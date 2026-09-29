#!/usr/bin/env python3
"""Plan and, when explicitly requested, build the environments for a profile.

The command line is deliberately safe by default.  It reads a profile and
prints a plan.  The build path is also available as ``run_bootstrap`` for a
Claude Science ``repl`` cell, where the pre-bound ``host`` object can be passed
explicitly.  This follows the same host boundary as ``dispatch_modal.py``:
ordinary command-line code does not pretend that the host object is importable.

Build results are append-only JSONL records.  Resolved values are emitted in a
separate JSON document and are never written back to the profile.

An environment is named to the host in one of two ways, and the plan says which
before anything is built.  The host loads a bare name from its own ``envs/``
directory.  A recipe this package ships has no file there, so it is named by
``path=`` instead.  ``self_shipped_recipe`` answers which by reading the
package's own ``envs/`` directory.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


UNRESOLVED = "__REQUIRED__"
MODAL_IMAGE_ENVIRONMENT_KEY = "CLAUDE_BINDER_MODAL_IMAGE"
"""Environment key `adapters/modal_platform.py` reads the image from (`IMAGE_KEY`)."""
DEFAULT_LEDGER = Path("bootstrap-environments.jsonl")
ENVIRONMENT_IDENTITY_RE = re.compile(
    r"^modal-env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)@spec_sha=(?P<spec_sha>[^@]+)$"
)
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
#: Where this package keeps the Modal recipes it ships itself.  `scripts/` and
#: `envs/` are siblings under `skills/claude-binder-lane/` in this repository
#: and under the skill root once installed, because the build flattens the two,
#: so this one expression is right in both layouts.
SELF_SHIPPED_ENVS_DIR = Path(__file__).resolve().parent.parent / "envs"


def shipped_environments(envs_dir: Path | None = None) -> list[str]:
    """Return the environment names this package ships a recipe for.

    The leading-underscore skip matches the host's own ``list_envs()``, which
    ignores a file whose name starts with one.
    """

    directory = SELF_SHIPPED_ENVS_DIR if envs_dir is None else Path(envs_dir)
    if not directory.is_dir():
        return []
    return sorted(
        path.stem
        for path in directory.glob("*.py")
        if not path.name.startswith("_")
    )


def self_shipped_recipe(environment: str, *, envs_dir: Path | None = None) -> Path | None:
    """Return this package's own recipe file for ``environment``, or ``None``.

    The host reads recipes only from its own ``envs/`` directory.  A bare name
    resolves for the environments the host ships and cannot resolve for the two
    this package ships, because the host holds no file by either name.  The
    platform's bring-up reference gives ``path=`` as the form for exactly that
    case at ``remote-compute-modal/env-setup.md`` line 186,
    ``build_env('proteomics_gpu_tx448', path='./proteomics_gpu_tx448.py')``, and
    the kernel signature is ``build_env(name, *, path=None, secrets=None,
    hydrate=False)`` loading ``path or f'{_ENVS}/{name}.py'``.

    The answer comes from reading the shipped directory, never from a list
    written here.  A list would name the recipes that exist today and go stale
    the next time one is added or removed.  That is the failure
    the package's own reproduction-readiness module already records for a
    hand-maintained answer in this package, and ``dispatch_modal.py``'s
    ``SHIPPED_GPU_DEFAULTS`` is a second hand-maintained list of the same names.
    Reading the directory costs one ``is_file`` call per environment and is
    correct by construction.
    """

    if not ENVIRONMENT_NAME_RE.match(environment):
        return None
    directory = SELF_SHIPPED_ENVS_DIR if envs_dir is None else Path(envs_dir)
    candidate = directory / f"{environment}.py"
    return candidate if candidate.is_file() else None


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_config_path(path: Path, reference: str) -> Path:
    candidate = Path(reference)
    return candidate if candidate.is_absolute() else path.parent / candidate


def load_profile(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a profile with the repository's base-profile and overlay rules.

    The profile files are templates.  A build must plan against the effective
    adapter list, so this mirrors the loader used by the lane package instead
    of looking only at the child file's ``overlay`` block.
    """

    resolved_path = Path(path).resolve()
    visited = set() if seen is None else set(seen)
    if resolved_path in visited:
        raise ValueError(f"profile inheritance cycle: {resolved_path}")
    visited.add(resolved_path)

    profile = load_json(resolved_path)
    if not isinstance(profile, dict):
        raise ValueError(f"profile must be a JSON object: {resolved_path}")

    base_ref = profile.get("base_profile")
    if not base_ref:
        return profile

    base = load_profile(_resolve_config_path(resolved_path, str(base_ref)), visited)
    overlay = profile.get("overlay", {})
    if not isinstance(overlay, dict):
        raise ValueError("profile overlay must be an object")

    merged = copy.deepcopy(base)
    top_level = overlay.get("top_level", {})
    if not isinstance(top_level, dict):
        raise ValueError("profile overlay.top_level must be an object")
    for key, value in top_level.items():
        merged[key] = copy.deepcopy(value)

    remove_adapters = set(overlay.get("remove_adapters", []))
    adapters = merged.get("adapters", [])
    if not isinstance(adapters, list):
        raise ValueError("profile adapters must be a list")
    merged["adapters"] = [
        adapter
        for adapter in adapters
        if isinstance(adapter, dict)
        and adapter.get("adapter_id") not in remove_adapters
    ]
    adapter_map = {
        adapter["adapter_id"]: adapter
        for adapter in merged["adapters"]
        if isinstance(adapter, dict) and "adapter_id" in adapter
    }

    defaults = overlay.get("adapter_defaults", {})
    if defaults:
        if not isinstance(defaults, dict):
            raise ValueError("profile overlay.adapter_defaults must be an object")
        for adapter in adapter_map.values():
            adapter.update(copy.deepcopy(defaults))

    adapter_overrides = overlay.get("adapter_overrides", [])
    if not isinstance(adapter_overrides, list):
        raise ValueError("profile overlay.adapter_overrides must be a list")
    for override in adapter_overrides:
        if not isinstance(override, dict) or "adapter_id" not in override:
            raise ValueError("each adapter override must name adapter_id")
        adapter_id = override["adapter_id"]
        if adapter_id not in adapter_map:
            adapter_map[adapter_id] = {}
            merged["adapters"].append(adapter_map[adapter_id])
        adapter_map[adapter_id].update(copy.deepcopy(override))

    stages = merged.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("profile stages must be a list")
    remove_stages = set(overlay.get("remove_stages", []))
    merged["stages"] = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage_id") not in remove_stages
    ]
    stage_map = {
        stage["stage_id"]: stage
        for stage in merged["stages"]
        if isinstance(stage, dict) and "stage_id" in stage
    }
    stage_overrides = overlay.get("stage_overrides", [])
    if not isinstance(stage_overrides, list):
        raise ValueError("profile overlay.stage_overrides must be a list")
    for override in stage_overrides:
        if not isinstance(override, dict) or "stage_id" not in override:
            raise ValueError("each stage override must name stage_id")
        stage_id = override["stage_id"]
        if stage_id not in stage_map:
            stage_map[stage_id] = {}
            merged["stages"].append(stage_map[stage_id])
        stage_map[stage_id].update(copy.deepcopy(override))

    campaign_overrides = overlay.get("campaign_overrides")
    if campaign_overrides is not None:
        merged["campaign_overrides"] = copy.deepcopy(campaign_overrides)
    return merged


def _path_text(parts: Iterable[str | int]) -> str:
    result = ""
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        elif not result:
            result = str(part)
        else:
            result += f".{part}"
    return result


def _walk_json(value: Any, parts: tuple[str | int, ...] = ()) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, parts + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, parts + (index,))
    else:
        yield _path_text(parts), value


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return repr(value)


def _identity_parts(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = ENVIRONMENT_IDENTITY_RE.fullmatch(value)
    if match is None:
        return None
    return match.group("name"), match.group("spec_sha")


def _target(
    *,
    adapter_index: int,
    adapter_id: str,
    environment: str,
    field: str,
    json_path: str,
    current: Any,
    needs_fill: bool,
    requires_hydration: bool = False,
) -> dict[str, Any]:
    return {
        "adapter_id": adapter_id,
        "environment": environment,
        "field": field,
        "json_path": json_path,
        "adapter_index": adapter_index,
        "current": _json_safe(current),
        "needs_fill": needs_fill,
        "requires_hydration": requires_hydration,
    }


def plan_profile(
    profile: dict[str, Any],
    *,
    hydrate: bool = False,
    envs_dir: Path | None = None,
) -> dict[str, Any]:
    """Return the ordered environment plan and its exact effective JSON paths.

    Each entry also carries how the environment will be named to the host.
    ``recipe_source`` is ``"package"`` when this package ships the recipe and
    ``"host"`` when the host owns it, and ``recipe_path`` is the file a
    ``"package"`` row will be built from.  The build reads that decision back
    rather than making it again, so a dry run states exactly what a build does.
    """

    adapters = profile.get("adapters", [])
    if not isinstance(adapters, list):
        raise ValueError("profile adapters must be a list")

    environments: dict[str, dict[str, Any]] = {}
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            continue
        adapter_id = str(adapter.get("adapter_id", f"adapter-{index}"))
        identity = adapter.get("environment_identity")
        parts = _identity_parts(identity)
        if parts is None:
            continue
        environment, spec_sha = parts
        entry = environments.setdefault(
            environment,
            {
                "environment": environment,
                "adapter_ids": [],
                "targets": [],
            },
        )
        entry["adapter_ids"].append(adapter_id)

        identity_path = _path_text(("adapters", index, "environment_identity"))
        entry["targets"].append(
            _target(
                adapter_index=index,
                adapter_id=adapter_id,
                environment=environment,
                field="spec_sha",
                json_path=identity_path,
                current=identity,
                needs_fill=spec_sha == UNRESOLVED,
            )
        )

        resources = adapter.get("resources")
        resources = resources if isinstance(resources, dict) else {}
        image_path = _path_text(
            ("adapters", index, "resources", "container_image_digest")
        )
        entry["targets"].append(
            _target(
                adapter_index=index,
                adapter_id=adapter_id,
                environment=environment,
                field="image",
                json_path=image_path,
                current=resources.get("container_image_digest", UNRESOLVED),
                needs_fill=resources.get("container_image_digest") == UNRESOLVED
                or "container_image_digest" not in resources,
            )
        )

        # The Modal dispatcher reads the image from this environment key, not from
        # resources.container_image_digest, and both hold the same string: in the
        # completed Modal evidence each appears 220 times with the identical value.
        # Filling only the resources field left four env slots for a person to copy
        # by hand from a value this script had just written one line above.
        adapter_environment = adapter.get("environment")
        adapter_environment = adapter_environment if isinstance(adapter_environment, dict) else {}
        if MODAL_IMAGE_ENVIRONMENT_KEY in adapter_environment:
            entry["targets"].append(
                _target(
                    adapter_index=index,
                    adapter_id=adapter_id,
                    environment=environment,
                    field="image",
                    json_path=_path_text(
                        ("adapters", index, "environment", MODAL_IMAGE_ENVIRONMENT_KEY)
                    ),
                    current=adapter_environment.get(MODAL_IMAGE_ENVIRONMENT_KEY, UNRESOLVED),
                    needs_fill=adapter_environment.get(MODAL_IMAGE_ENVIRONMENT_KEY) == UNRESOLVED,
                )
            )

        model_path = _path_text(("adapters", index, "model_revision"))
        model_revision = adapter.get("model_revision", UNRESOLVED)
        entry["targets"].append(
            _target(
                adapter_index=index,
                adapter_id=adapter_id,
                environment=environment,
                field="model_revision",
                json_path=model_path,
                current=model_revision,
                needs_fill=model_revision == UNRESOLVED,
                requires_hydration=True,
            )
        )

    ordered = []
    for entry in environments.values():
        entry = copy.deepcopy(entry)
        entry["targets"] = [
            target
            for target in entry["targets"]
            if target["field"] != "model_revision" or hydrate or target["needs_fill"]
        ]
        entry["would_fill"] = [
            target["json_path"]
            for target in entry["targets"]
            if target["needs_fill"]
            and (not target["requires_hydration"] or hydrate)
        ]
        entry["hydrate"] = bool(hydrate)
        recipe = self_shipped_recipe(entry["environment"], envs_dir=envs_dir)
        entry["recipe_path"] = str(recipe) if recipe is not None else None
        entry["recipe_source"] = "host" if recipe is None else "package"
        ordered.append(entry)
    return {"environments": ordered}


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid ledger JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"ledger row {line_number} is not an object")
        rows.append(row)
    return rows


def _append_ledger(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _successful_record(
    rows: Iterable[dict[str, Any]],
    environment: str,
    *,
    spec_sha: str | None,
    hydrate: bool,
) -> dict[str, Any] | None:
    for row in reversed(list(rows)):
        if row.get("environment") != environment or row.get("outcome") != "succeeded":
            continue
        if hydrate and row.get("hydrate_requested") is not True:
            continue
        result = row.get("result")
        if spec_sha is not None and (
            not isinstance(result, dict) or result.get("spec_sha") != spec_sha
        ):
            continue
        return row
    return None


def _classify_result(result: Any, *, hydrate: bool) -> str:
    if not isinstance(result, dict):
        return "build_failed"
    image_ok = isinstance(result.get("image"), str) and bool(result["image"])
    spec_ok = isinstance(result.get("spec_sha"), str) and bool(result["spec_sha"])
    if not image_ok or not spec_ok:
        return "build_failed"
    if result.get("hydrate_error"):
        return "hydrate_failed"
    if hydrate and result.get("hydrated") is not True:
        return "hydrate_failed"
    return "succeeded"


def _values_from_result(
    environment_plan: dict[str, Any],
    result: Any,
    *,
    outcome: str,
    hydrate: bool,
) -> list[dict[str, Any]]:
    if outcome not in {"succeeded", "hydrate_failed"} or not isinstance(result, dict):
        return []

    values: list[dict[str, Any]] = []
    spec_sha = result.get("spec_sha")
    image = result.get("image")
    model_revision = result.get("model_revision")
    for target in environment_plan["targets"]:
        value: Any = None
        if target["field"] == "spec_sha" and isinstance(spec_sha, str) and spec_sha:
            value = f"modal-env:{environment_plan['environment']}@spec_sha={spec_sha}"
        elif target["field"] == "image" and isinstance(image, str) and image:
            value = image
        elif (
            target["field"] == "model_revision"
            and hydrate
            and outcome == "succeeded"
            and result.get("hydrated") is True
            and isinstance(model_revision, str)
            and model_revision
        ):
            value = model_revision
        if value is None or not target["needs_fill"]:
            continue
        values.append(
            {
                "environment": environment_plan["environment"],
                "adapter_id": target["adapter_id"],
                "field": target["field"],
                "json_path": target["json_path"],
                "value": value,
            }
        )
    return values


def _remaining_required(
    profile: dict[str, Any], filled_paths: set[str]
) -> list[dict[str, Any]]:
    def reason(path: str, value: Any) -> str:
        if path.endswith(".model_revision"):
            return (
                "a hydrated weight digest is not available from this run; a "
                "person must record the digest after the weight volume exists"
            )
        if path.endswith(".environment_identity"):
            if isinstance(value, str) and value.startswith("modal-env:"):
                return "the environment build did not return a spec_sha for this field"
            return "the profile does not name an environment, so a person must choose one"
        if path.endswith(".container_image_digest"):
            return "the environment build did not return an image for this field"
        return "this field is outside the environment build result and needs a person"

    remaining: list[dict[str, Any]] = []
    for path, value in _walk_json(profile):
        if value != UNRESOLVED or path in filled_paths:
            continue
        remaining.append(
            {"json_path": path, "value": value, "reason": reason(path, value)}
        )

    # A spec hash is embedded in the identity string, so the scalar walker
    # cannot see that a successful build filled only its hash component.
    for index, adapter in enumerate(profile.get("adapters", [])):
        if not isinstance(adapter, dict):
            continue
        identity = adapter.get("environment_identity")
        parts = _identity_parts(identity)
        if parts is None or parts[1] != UNRESOLVED:
            continue
        path = _path_text(("adapters", index, "environment_identity"))
        if path in filled_paths:
            remaining = [row for row in remaining if row["json_path"] != path]
        elif not any(row["json_path"] == path for row in remaining):
            remaining.append(
                {"json_path": path, "value": identity, "reason": reason(path, identity)}
            )
    return sorted(remaining, key=lambda row: row["json_path"])


def _build_error_text(
    exc: BaseException,
    environment: str,
    recipe_path: str | None,
    envs_dir: Path | None,
) -> str:
    """Return the recorded error, naming the recipe route when no file was found.

    The bare message a missing recipe produces is
    ``FileNotFoundError: [Errno 2] No such file or directory:
    '<host envs dir>/<name>.py'``.  It names a directory inside the installed
    host runtime and says nothing about this package shipping recipes of its
    own, so the operator who meets it at bring-up has no way to tell a typo
    from an environment that needed ``path=``.
    """

    text = f"{type(exc).__name__}: {exc}"
    if not isinstance(exc, FileNotFoundError):
        return text
    if recipe_path is not None:
        return (
            f"{text}. {environment} was named by path, from this package's own "
            f"recipe at {recipe_path}, and the kernel that ran the build could "
            "not read that file"
        )
    directory = SELF_SHIPPED_ENVS_DIR if envs_dir is None else Path(envs_dir)
    shipped = ", ".join(shipped_environments(envs_dir)) or "none"
    return (
        f"{text}. {environment} was named to the host bare, which the host "
        f"loads from its own envs directory, because this package ships no "
        f"{environment}.py. The recipes this package ships are: {shipped}. "
        f"Either the host has no environment called {environment}, or this "
        f"package owes one at {directory / f'{environment}.py'}"
    )


def run_bootstrap(
    profile_path: Path,
    *,
    host: Any | None = None,
    build: bool = False,
    hydrate: bool = False,
    ledger_path: Path | None = None,
    output_path: Path | None = None,
    envs_dir: Path | None = None,
) -> dict[str, Any]:
    """Plan or execute the environment bootstrap.

    ``build=False`` is the safe default and does not inspect or call ``host``.
    For a real build, pass the caller-owned host explicitly.  The sole host
    call is ``host.build_env(name, hydrate=hydrate)``, with ``path=`` added for
    an environment this package ships its own recipe for.  No budget, ceiling,
    timeout, or secret is supplied by this script.
    """

    if hydrate and not build:
        raise ValueError("--hydrate requires the explicit --build flag")
    profile_path = Path(profile_path).resolve()
    profile = load_profile(profile_path)
    plan = plan_profile(profile, hydrate=hydrate, envs_dir=envs_dir)
    ledger = Path(ledger_path) if ledger_path is not None else DEFAULT_LEDGER
    rows = _load_ledger(ledger)

    if output_path is not None and Path(output_path).resolve() == profile_path:
        raise ValueError("the output JSON document cannot be the profile itself")
    if build and host is None:
        raise RuntimeError(
            "a real build needs the caller's pre-bound host; import this script "
            "inside the repl cell and pass host=host to run_bootstrap"
        )

    outcomes: list[dict[str, Any]] = []
    resolved_values: list[dict[str, Any]] = []
    for environment_plan in plan["environments"]:
        environment = environment_plan["environment"]
        recipe_hashes = {
            parts[1]
            for target in environment_plan["targets"]
            if target["field"] == "spec_sha"
            for parts in [_identity_parts(target["current"])]
            if parts is not None and parts[1] != UNRESOLVED
        }
        requested_spec_sha = next(iter(recipe_hashes)) if len(recipe_hashes) == 1 else None
        prior = _successful_record(
            rows,
            environment,
            spec_sha=requested_spec_sha,
            hydrate=hydrate,
        )
        if prior is not None:
            result = prior.get("result")
            outcome = "skipped_succeeded"
            resolved_values.extend(
                _values_from_result(
                    environment_plan,
                    result,
                    outcome="succeeded",
                    hydrate=hydrate,
                )
            )
            outcomes.append(
                {
                    "environment": environment,
                    "outcome": outcome,
                    "result": _json_safe(result),
                    "ledger_recorded_at": prior.get("recorded_at"),
                }
            )
            continue

        if not build:
            outcomes.append(
                {
                    "environment": environment,
                    "outcome": "planned",
                    "hydrate_requested": hydrate,
                    "result": None,
                }
            )
            continue

        assert host is not None
        result: Any = None
        error: str | None = None
        recipe_path = environment_plan.get("recipe_path")
        try:
            build_env = getattr(host, "build_env", None)
            if not callable(build_env):
                raise TypeError("host has no callable build_env(name, hydrate=...) method")
            # Do not add a budget or ceiling here.  The caller owns that choice.
            build_arguments: dict[str, Any] = {"hydrate": hydrate}
            if recipe_path is not None:
                # A bare name loads `_ENVS/<name>.py` from the host's own envs
                # directory, which holds no file for a recipe this package
                # ships.  `path=` is the documented form for that case.
                build_arguments["path"] = recipe_path
            result = build_env(environment, **build_arguments)
            outcome = _classify_result(result, hydrate=hydrate)
        except Exception as exc:
            outcome = "build_failed"
            error = _build_error_text(exc, environment, recipe_path, envs_dir)

        row = {
            "record": "environment_build",
            "environment": environment,
            "outcome": outcome,
            # Which recipe file the image came from. A cached image a later
            # session reuses is only reproducible if the recipe behind it is
            # identifiable, and a bare name does not identify one once this
            # package ships recipes of its own.
            "recipe_path": recipe_path,
            "hydrate_requested": hydrate,
            "result": _json_safe(result),
            "error": error,
            "recorded_at": utc_now(),
        }
        _append_ledger(ledger, row)
        rows.append(row)
        resolved_values.extend(
            _values_from_result(
                environment_plan,
                result,
                outcome=outcome,
                hydrate=hydrate,
            )
        )
        outcomes.append(row)

    filled_paths = {item["json_path"] for item in resolved_values}
    remaining = _remaining_required(profile, filled_paths)
    report = {
        "profile": str(profile_path),
        "dry_run": not build,
        "hydrate_requested": hydrate,
        "plan": plan["environments"],
        "outcomes": _json_safe(outcomes),
        "resolved": _json_safe(resolved_values),
        "unresolved_required": remaining,
        "unresolved_paths": [row["json_path"] for row in remaining],
        "ledger": str(ledger),
        "profile_write": "none",
        "profile_write_reason": (
            "the build result is evidence for a person or later apply step; "
            "writing it here would mutate the campaign input before review"
        ),
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _cli_host() -> Any | None:
    """Return an injected repl host if this module was executed in that scope."""

    return globals().get("host")


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan Modal environment builds. Dry run is the default. Real builds "
            "require --build and a caller-owned host passed to run_bootstrap."
        )
    )
    parser.add_argument("profile", type=Path)
    parser.add_argument(
        "--build",
        action="store_true",
        help="execute builds through an explicitly supplied repl host",
    )
    parser.add_argument(
        "--hydrate",
        action="store_true",
        help="request the explicit build_env(hydrate=True) weight step",
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the separate JSON report here instead of only printing it",
    )
    args = parser.parse_args(argv)
    try:
        report = run_bootstrap(
            args.profile,
            host=_cli_host() if args.build else None,
            build=args.build,
            hydrate=args.hydrate,
            ledger_path=args.ledger,
            output_path=args.output,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
