"""Choose how a packaged fal client is invoked.

Every fal client in this package reads its API credential from a named process
environment variable. The default name is ``FAL_KEY``. The
``CLAUDE_BINDER_FAL_CREDENTIAL_ENV`` selector can name a host-provided alias
without copying the credential value. Two routes expose it.

The **direct** route runs the client as a child of this process. It works when
this process already carries the selected credential variable.

The **wrapper** route runs the client under a local credential launcher. The
launcher reads the credential from an operating system credential store and
exports it into the child. On the workstation this project was built on, that
launcher is ``fal-credential-wrapper`` and the subcommand is ``exec-model``. Its whole job is
to export ``FAL_KEY`` and then exec the command after ``--``.

Neither route is available everywhere. A Claude Science session has no
launcher. A workstation that keeps the credential out of every shell
environment has no ``FAL_KEY``. So the route is chosen per call, from what the
environment actually offers.

Nothing here returns, logs, or compares a credential value. The only question
asked of the selected variable is whether it holds a non-empty string.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path


CREDENTIAL_ENVIRONMENT_KEY = "FAL_KEY"
CREDENTIAL_ENVIRONMENT_OVERRIDE = "CLAUDE_BINDER_FAL_CREDENTIAL_ENV"
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
ROUTE_ENVIRONMENT_KEY = "CLAUDE_BINDER_FAL_ROUTE"
WRAPPER_SUBCOMMAND = "exec-model"

ROUTE_AUTO = "auto"
ROUTE_DIRECT = "direct"
ROUTE_WRAPPER = "wrapper"
ROUTES = (ROUTE_AUTO, ROUTE_DIRECT, ROUTE_WRAPPER)
REDACTED_FAL_URL = "https://fal.run/<account>/<application>"


def redacted_argv(argv: Iterable[str]) -> list[str]:
    """Copy an argv for logs without exposing a configured fal endpoint."""
    values = [str(value) for value in argv]
    for index, value in enumerate(values[:-1]):
        if value == "--fal-url":
            values[index + 1] = REDACTED_FAL_URL
    return values


def redacted_command(argv: Iterable[str]) -> str:
    """Render a redacted argv for a human-readable log line."""
    return shlex.join(redacted_argv(argv))


class RouteError(RuntimeError):
    """No usable route puts the fal credential in front of a client."""


def credential_environment_key(
    explicit: str | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Return a validated variable name. Never read or expose its value."""
    source = os.environ if environ is None else environ
    selected = (
        explicit
        if explicit is not None
        else source.get(CREDENTIAL_ENVIRONMENT_OVERRIDE, "")
        or CREDENTIAL_ENVIRONMENT_KEY
    )
    value = str(selected).strip()
    if ENVIRONMENT_NAME_RE.fullmatch(value) is None:
        raise RouteError(
            "the fal credential environment selector must be a variable name such as FAL_KEY"
        )
    return value


def credential_present(
    environ: Mapping[str, str] | None = None,
    *,
    credential_env_key: str | None = None,
) -> bool:
    """Report whether the fal credential is in the environment.

    A true result proves only credential presence. It does not prove access to
    any configured application. The provider authorization preflight tests that
    access before a paid stage can run. The value never leaves this call.
    """
    source = os.environ if environ is None else environ
    key = credential_environment_key(credential_env_key, source)
    return bool(str(source.get(key, "") or "").strip())


def wrapper_path(executable: str | os.PathLike[str] | None) -> str | None:
    """Return the runnable credential wrapper, or None when there is none.

    A bare name is looked up on PATH. A name with a directory component is
    checked where it points.
    """
    if not executable:
        return None
    return shutil.which(os.path.expanduser(str(executable)))


def requested_route(
    explicit: str | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """Return the route the caller asked for, defaulting to automatic.

    An explicit choice wins. ``CLAUDE_BINDER_FAL_ROUTE`` is the fallback,
    because the shipped profiles pass no route flag and cannot be edited from
    a running session.
    """
    source = os.environ if environ is None else environ
    value = str(explicit or source.get(ROUTE_ENVIRONMENT_KEY, "") or "").strip().lower()
    if not value:
        return ROUTE_AUTO
    if value not in ROUTES:
        raise RouteError(
            f"unknown fal credential route {value!r}. "
            f"Choose one of {', '.join(ROUTES)}."
        )
    return value


def _both_routes_sentence(executable: str) -> str:
    """Return the sentence pair that names each route and what it needs."""
    return (
        f"Set {CREDENTIAL_ENVIRONMENT_KEY} in this process's environment and the "
        "adapter calls the client directly. Install the credential wrapper "
        f"{executable!r} on PATH and the adapter calls the client through "
        f"{executable} {WRAPPER_SUBCOMMAND}, which supplies the credential itself. "
        f"A present {CREDENTIAL_ENVIRONMENT_KEY} proves only presence. The provider "
        "authorization preflight verifies application access."
    )


def resolve_route(
    executable: str,
    *,
    requested: str | None = None,
    environ: Mapping[str, str] | None = None,
    credential_env_key: str | None = None,
) -> str:
    """Return the route this call takes, or raise an error naming both routes."""
    choice = requested_route(requested, environ)
    key = credential_environment_key(credential_env_key, environ)
    has_credential = credential_present(
        environ, credential_env_key=key
    )
    has_wrapper = wrapper_path(executable) is not None

    if choice == ROUTE_DIRECT:
        if has_credential:
            return ROUTE_DIRECT
        raise RouteError(
            f"the direct fal route was requested, and {key} is "
            "absent from this process's environment. Set it, or pass "
            f"--fal-credential-route {ROUTE_WRAPPER} to use {executable!r} instead."
        )

    if choice == ROUTE_WRAPPER:
        if has_wrapper:
            return ROUTE_WRAPPER
        raise RouteError(
            f"the wrapper fal route was requested, and {executable!r} is not on PATH. "
            f"Install it, or set {CREDENTIAL_ENVIRONMENT_KEY} and pass "
            f"--fal-credential-route {ROUTE_DIRECT} instead."
        )

    if has_credential:
        return ROUTE_DIRECT
    if has_wrapper:
        return ROUTE_WRAPPER
    raise RouteError(
        "no route to the fal credential is available, so the client cannot run. "
        f"Two routes exist. {_both_routes_sentence(executable)} "
        f"Right now {CREDENTIAL_ENVIRONMENT_KEY} is absent from the environment and "
        f"{executable!r} is not on PATH."
    )


def client_command(
    executable: str,
    interpreter: str,
    client: str | os.PathLike[str],
    arguments: Iterable[str] = (),
    *,
    requested: str | None = None,
    environ: Mapping[str, str] | None = None,
    credential_env_key: str | None = None,
) -> list[str]:
    """Return the argument list that runs one packaged fal client.

    The direct route drops the launcher and its separator. Everything after
    them is identical on both routes, so a client sees the same arguments
    either way.
    """
    tail = [str(interpreter), str(Path(client)), *(str(value) for value in arguments)]
    route = resolve_route(
        executable,
        requested=requested,
        environ=environ,
        credential_env_key=credential_env_key,
    )
    if route == ROUTE_DIRECT:
        return tail
    return [str(executable), WRAPPER_SUBCOMMAND, "--", *tail]


def add_route_argument(parser: argparse.ArgumentParser, *, executable: str) -> None:
    """Add the route override every fal adapter accepts.

    The flag names a route. It never accepts a credential, because an argument
    is visible in a process listing.
    """
    parser.add_argument(
        "--fal-credential-route",
        choices=ROUTES,
        default=None,
        help=(
            f"How the fal client receives {CREDENTIAL_ENVIRONMENT_KEY}. "
            f"{ROUTE_DIRECT} runs the client in this environment. "
            f"{ROUTE_WRAPPER} runs it through {executable} {WRAPPER_SUBCOMMAND}. "
            f"{ROUTE_AUTO} prefers direct when {CREDENTIAL_ENVIRONMENT_KEY} is set "
            f"and falls back to the wrapper. Defaults to {ROUTE_ENVIRONMENT_KEY}, "
            f"then to {ROUTE_AUTO}."
        ),
    )


def add_credential_environment_argument(parser: argparse.ArgumentParser) -> None:
    """Add an environment-variable name selector. Credential values stay out of argv."""
    parser.add_argument(
        "--fal-credential-env",
        type=credential_environment_key,
        default=None,
        help=(
            "Environment-variable name holding the fal credential for the direct route. "
            f"Defaults to {CREDENTIAL_ENVIRONMENT_OVERRIDE}, then {CREDENTIAL_ENVIRONMENT_KEY}."
        ),
    )
