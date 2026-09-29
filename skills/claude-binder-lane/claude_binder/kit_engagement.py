#!/usr/bin/env python3
"""Decide whether an accelerated kit really engaged, from what it printed.

The 2026-09-17 optimization kits wrap an unmodified upstream model and can run
it four ways. ``off`` is stock as published, ``exact`` reproduces the unmodified
model, ``fast`` is the package default, and ``big`` lowers peak memory. A kit
announces which one took effect and refuses rather than degrading quietly.

**The whole value of a kit is the mode, and a caller that does not read the
announcement cannot tell an accelerated run from a stock one.** Both cost the
same GPU minutes. Both write the same kind of output. A run that asked for
``fast`` and quietly served stock bytes is indistinguishable from a successful
one unless something parses the lines below, which is what this module is for.

The contract, read from `boltz2/run.sh` and `boltz2/README.md` at release commit
f4f62fa6592ae4938d49b1757bea0cfeff9f468e of
github.com/anthropics/uplifting-biomolecular-modeling:

- An engaged mode prints ``[<kit>-opt] ACTIVE mode=<mode> ... levers=...`` and
  ends with one ``LEVER name=... state=...`` per optimization and
  ``[<kit>-opt] EXIT mode=<mode> ... rc=<rc>``.
- A mode that cannot engage prints ``[<kit>-opt] NOT ACTIVE: <reason>`` and
  exits 3. The kit states it never falls back to stock on its own.
- Kits do not share one punctuation for that line. boltz2 writes
  ``NOT ACTIVE: <reason>``, colabfold writes
  ``NOT ACTIVE mode=off (stock: nothing applied)``, af3_torch writes
  ``NOT ACTIVE mode=<mode> reason=<reason>``, and one kit brackets the mode.
  Requiring the colon drops three of those four, so the colon is optional here
  and a ``mode=`` or ``reason=`` tail is read out when it is present.
- ``check`` can print ``[<kit>-opt] DRY-RUN mode=<mode> ...``. This is
  readiness evidence, not evidence that a prediction engaged.
- An ACTIVE line says which build started, not that the run may be attributed
  to it. A weights digest mismatch exits 1, a usage rejection exits 2 and a
  kernel census fallback exits 5, and each of those removes the basis for the
  attribution whatever the announcement said.
- A lever the card does not support is named on the ACTIVE line as
  ``card_off=<lever>:<reason>``, and the mode still counts as engaged.
- Partial activation is a mode that engaged without every optimization. Kits
  report it as ``NOT ACTIVE: partial activation - <detail>``, as
  ``PARTIAL refused=<kind> rc=3`` after the outputs are written, and as
  ``partial=<ids>`` on a summary line. It exits 3 unless ``--allow-partial`` was
  passed, and boltz2 calls that flag "the one deliberate exception, and it says
  which optimization it left out". So a partial run under the flag is an
  accelerated run that the kit itself accepted, and this module accepts it too.
  ``(partial: none)`` states the opposite and does not count.

**The trap this module exists to catch.** ``--mode off`` is a legitimate request
and it also prints ``NOT ACTIVE: mode off: stock ...``. So "NOT ACTIVE" alone
means nothing. A parser that treats every NOT ACTIVE as a failure rejects
deliberate stock runs, and a parser that treats every exit 0 as success accepts
a run that asked for ``fast`` and got stock. The verdict below is a comparison
between what was asked for and what engaged, never a scan for one string.

This module reads text and returns a verdict. It launches nothing and costs
nothing, so it runs on a machine with no GPU, and the same parser serves a live
run and a recorded log.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

KIT_MODES = ("off", "exact", "fast", "big")
ACCELERATED_MODES = ("exact", "fast", "big")

# `run.sh` exit codes, and the two kit-specific ones the release documents.
EXIT_OK = 0
EXIT_WEIGHTS_DIGEST_MISMATCH = 1
EXIT_USAGE = 2
EXIT_NOT_ACTIVE = 3
EXIT_KERNEL_CENSUS_FALLBACK = 5

EXIT_MEANINGS = {
    EXIT_OK: "the command completed",
    EXIT_WEIGHTS_DIGEST_MISMATCH: "the command failed or was incomplete (for example, a weights digest mismatch)",
    EXIT_USAGE: "the command line was rejected before anything ran",
    EXIT_NOT_ACTIVE: "the requested mode did not engage",
    EXIT_KERNEL_CENSUS_FALLBACK: "the kernel census found a fallback where a kit kernel was required",
}

_ACTIVE = re.compile(r"\[(?P<kit>[\w.-]+)-opt\]\s+ACTIVE\s+mode=(?P<mode>\w+)(?P<rest>.*)")
_ACTIVE_LEVERS = re.compile(r"\blevers=(?P<names>[\w.,:-]+)")
_NOT_ACTIVE = re.compile(r"\[(?P<kit>[\w.-]+)-opt\]\s+NOT ACTIVE\b:?\s*(?P<reason>.*)")
# The stock arm prints this without the bracketed prefix in at least one path,
# so the prefix is optional here and required nowhere else.
_NOT_ACTIVE_LOOSE = re.compile(r"NOT ACTIVE\b:?\s*(?P<reason>.*)")
# Read out of a NOT ACTIVE tail written in key=value form.
_TAIL_MODE = re.compile(r"\bmode=(?P<mode>\w+)")
_TAIL_REASON = re.compile(r"\breason=(?P<reason>.+)")
_EXIT = re.compile(r"\[(?P<kit>[\w.-]+)-opt\]\s+EXIT\s+mode=(?P<mode>\w+).*?\brc=(?P<rc>-?\d+)")
_LEVER = re.compile(r"\bLEVER\s+name=(?P<name>[\w.-]+)\s+state=(?P<state>[\w.:-]+)")
_LEVER_REASON = re.compile(r"\breason=(?P<reason>[^\s]+)")
_CARD_OFF = re.compile(r"\bcard_off=(?P<lever>[\w.-]+):(?P<reason>[^\s]+)")
_DRY_RUN = re.compile(r"\[(?P<kit>[\w.-]+)-opt\]\s+DRY-RUN\s+mode=(?P<mode>\w+)(?P<rest>.*)")
_DRY_RUN_REFUSAL = re.compile(r"\bwould_refuse=")
_STOCK_REASON = re.compile(r"^mode[ =]off\b")
_PARTIAL_REASON = re.compile(r"^partial activation\b", re.IGNORECASE)
_PARTIAL_LINE = re.compile(r"\bPARTIAL\s+refused=(?P<kind>[\w.:-]+)")
# The lookbehind keeps an echoed `--allow-partial=1` out of this. The flag name
# ends in the same word, and a log that repeats the command line would otherwise
# read as evidence of the thing the flag exists to permit.
_PARTIAL_KV = re.compile(r"(?<![-\w])partial=(?!none\b)(?P<detail>[\w.,:-]+)")

# States a verdict can carry. `stock_by_request` is a success and every other
# non-active state is not.
ACTIVE = "active"
ACTIVE_WITH_CARD_LEVERS_OFF = "active-with-card-levers-off"
STOCK_BY_REQUEST = "stock-by-request"
MODE_MISMATCH = "mode-mismatch"
UNATTRIBUTABLE_EXIT = "unattributable-exit"
ACTIVE_PARTIAL = "active-partial"
NOT_ACTIVE = "not-active"
SILENT_STOCK = "silent-stock"
NO_ANNOUNCEMENT = "no-announcement"
DRY_RUN_READY = "dry-run-ready"


class KitEngagementError(ValueError):
    """The kit output cannot be read as an honest statement about its mode."""


def _read_not_active_tail(tail: str, mode: str | None) -> tuple[str, str | None]:
    """Split a NOT ACTIVE tail into its reason and the mode it names, if any."""
    tail = tail.strip()
    declared = _TAIL_MODE.search(tail)
    if declared and mode is None:
        mode = declared.group("mode")
    spelled = _TAIL_REASON.search(tail)
    if spelled:
        return spelled.group("reason").strip(), mode
    return tail, mode


def _lines(*streams: str) -> list[str]:
    out: list[str] = []
    for stream in streams:
        if stream:
            out.extend(stream.splitlines())
    return out


def parse_engagement(
    *,
    requested_mode: str,
    stdout: str = "",
    stderr: str = "",
    exit_code: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Return what the kit said about its own mode, and whether it matches the request."""
    if requested_mode not in KIT_MODES:
        raise KitEngagementError(
            f"{requested_mode!r} is not a kit mode; the kit offers {', '.join(KIT_MODES)}"
        )
    lines = _lines(stdout, stderr)

    kit_id: str | None = None
    active_mode: str | None = None
    active_lever_names: set[str] = set()
    levers: list[dict[str, str]] = []
    invalid_levers: list[str] = []
    card_off: list[dict[str, str]] = []
    not_active_reason: str | None = None
    not_active_mode: str | None = None
    partial_signals: list[str] = []
    exit_line_mode: str | None = None
    exit_line_rc: int | None = None
    dry_run_mode: str | None = None
    dry_run_refused = False

    for line in lines:
        if _PARTIAL_LINE.search(line) or _PARTIAL_KV.search(line):
            partial_signals.append(line.strip())
        match = _ACTIVE.search(line)
        if match:
            kit_id = kit_id or match.group("kit")
            active_mode = match.group("mode")
            named_levers = _ACTIVE_LEVERS.search(match.group("rest"))
            if named_levers and not named_levers.group("names").isdigit():
                active_lever_names.update(named_levers.group("names").split(","))
            for card in _CARD_OFF.finditer(match.group("rest")):
                card_off.append(
                    {"lever": card.group("lever"), "reason": card.group("reason")}
                )
            continue
        match = _NOT_ACTIVE.search(line)
        if match:
            kit_id = kit_id or match.group("kit")
            not_active_reason, not_active_mode = _read_not_active_tail(
                match.group("reason"), not_active_mode
            )
            continue
        match = _NOT_ACTIVE_LOOSE.search(line)
        if match and not_active_reason is None:
            not_active_reason, not_active_mode = _read_not_active_tail(
                match.group("reason"), not_active_mode
            )
            continue
        match = _EXIT.search(line)
        if match:
            kit_id = kit_id or match.group("kit")
            exit_line_mode = match.group("mode")
            exit_line_rc = int(match.group("rc"))
            continue
        match = _DRY_RUN.search(line)
        if match:
            kit_id = kit_id or match.group("kit")
            dry_run_mode = match.group("mode")
            dry_run_refused = bool(_DRY_RUN_REFUSAL.search(match.group("rest")))
            continue
        for lever in _LEVER.finditer(line):
            reason_match = _LEVER_REASON.search(line[lever.end():])
            reason = reason_match.group("reason") if reason_match else None
            item = {"name": lever.group("name"), "state": lever.group("state")}
            if reason:
                item["reason"] = reason
            levers.append(item)
            state_name = lever.group("state").lower()
            if state_name not in {"on", "off", "served", "skipped"} or (
                state_name == "skipped"
                and (not reason or re.match(r"^(?:fallback|refused|error)(?::|$)", reason))
            ):
                invalid_levers.append(lever.group("name"))

    invalid_levers.extend(
        item["name"] for item in levers
        if item["name"] in active_lever_names and item["state"].lower() == "off"
    )

    if dry_run_mode is not None and active_mode is None and not_active_reason is None:
        if dry_run_mode != requested_mode:
            state, detail = (
                NOT_ACTIVE,
                f"the dry run checked {dry_run_mode}, not the requested {requested_mode} mode",
            )
        elif dry_run_refused:
            state, detail = (NOT_ACTIVE, f"the {requested_mode} dry run would refuse this card")
        elif exit_code == EXIT_OK:
            state, detail = (
                DRY_RUN_READY,
                f"the {requested_mode} check completed; no prediction ran or engaged",
            )
        else:
            state, detail = (
                NOT_ACTIVE,
                f"the {requested_mode} check did not complete successfully "
                f"(exit code {exit_code if exit_code is not None else 'missing'})",
            )
    else:
        state, detail = _classify(
            requested_mode=requested_mode,
            active_mode=active_mode,
            not_active_mode=not_active_mode,
            not_active_reason=not_active_reason,
            partial=bool(partial_signals),
            exit_code=exit_code,
            allow_partial=allow_partial,
        )
    # A card the kit configures but did not measure on switches levers off and
    # still engages. On A100, `exact` drops `msa_pwa_exact` this way. The mode
    # did run, so this stays accepted, and the state says the build was not the
    # one the published figures describe.
    if state == ACTIVE and card_off:
        state = ACTIVE_WITH_CARD_LEVERS_OFF
        names = ", ".join(sorted(item["lever"] for item in card_off))
        detail = (
            f"{active_mode} engaged with {len(card_off)} lever(s) switched off for this "
            f"card: {names}. Published timings for this mode were measured with them on."
        )
    # The announcement says which build started, not that the run may be
    # attributed to it. These exit codes remove that basis whatever the
    # announcement said, so `engaged` keeps the announcement and the state
    # refuses. Reading the ACTIVE line alone accepts a run whose weights were
    # not the published ones.
    if state in (ACTIVE, ACTIVE_WITH_CARD_LEVERS_OFF, STOCK_BY_REQUEST, ACTIVE_PARTIAL) and (
        exit_code is None
        or exit_code != EXIT_OK
        or (exit_line_rc is not None and exit_line_rc != exit_code)
        or (exit_line_mode is not None and exit_line_mode != requested_mode)
    ):
        state = UNATTRIBUTABLE_EXIT
        if exit_code is None:
            detail = "the process exit code is missing, so completion cannot be verified"
        elif exit_code != EXIT_OK:
            detail = (
                f"the process exited {exit_code}: "
                f"{EXIT_MEANINGS.get(exit_code, 'the run did not complete successfully')}. "
                "The run cannot be attributed to a completed build."
            )
            if exit_line_rc is not None and exit_line_rc != exit_code:
                detail += f" The kit's EXIT line also claims rc={exit_line_rc}."
        elif exit_line_rc is not None and exit_line_rc != exit_code:
            detail = f"the process exited {exit_code}, but the kit's EXIT line claims rc={exit_line_rc}"
        elif exit_line_mode is not None and exit_line_mode != requested_mode:
            detail = f"the kit's EXIT line names {exit_line_mode}, not the requested {requested_mode} mode"
    if state in (ACTIVE, ACTIVE_WITH_CARD_LEVERS_OFF, ACTIVE_PARTIAL) and invalid_levers:
        state = UNATTRIBUTABLE_EXIT
        detail = (
            "the kit's LEVER lines report unserved or unrecognised states for "
            + ", ".join(sorted(set(invalid_levers)))
        )

    return {
        "schema_version": 1,
        "kit_id": kit_id,
        "requested_mode": requested_mode,
        "active_mode": active_mode,
        "state": state,
        "detail": detail,
        # A mode that engaged is engaged even when it is the wrong one or came
        # up short of a lever. Saying otherwise would misreport what ran on the
        # card, which is the one thing this module exists to get right.
        "engaged": (active_mode is not None and active_mode != "off")
        or state == ACTIVE_PARTIAL
        or bool(partial_signals)
        or bool(not_active_reason and _PARTIAL_REASON.match(not_active_reason)),
        "accepted": state in (ACTIVE, ACTIVE_WITH_CARD_LEVERS_OFF, STOCK_BY_REQUEST)
        or (state == ACTIVE_PARTIAL and allow_partial),
        "not_active_reason": not_active_reason,
        "not_active_mode": not_active_mode,
        "partial_signals": partial_signals,
        "levers": levers,
        "card_off": card_off,
        "dry_run_ok": state == DRY_RUN_READY,
        "dry_run_mode": dry_run_mode,
        "exit_code": exit_code,
        "exit_code_meaning": EXIT_MEANINGS.get(exit_code) if exit_code is not None else None,
        "exit_line_mode": exit_line_mode,
        "exit_line_rc": exit_line_rc,
        "allow_partial": allow_partial,
    }


_STOCK_BY_REQUEST_DETAIL = "stock ran because the run asked for off, which is the stock arm"


def _partial_detail(mode: str, allow_partial: bool, reason: str | None) -> str:
    """Say what a partial activation was and whether the run allowed it."""
    named = f", which the kit reported as {reason}" if reason else ""
    if allow_partial:
        return (
            f"{mode} engaged without every optimization{named}, and the run passed "
            f"--allow-partial, which the kit treats as acceptance. Published timings "
            f"for this mode were measured with every optimization on."
        )
    return (
        f"{mode} engaged without every optimization{named}, and the run did not pass "
        f"--allow-partial, so the kit's own exit code refuses it"
    )


def _classify(
    *,
    requested_mode: str,
    active_mode: str | None,
    not_active_mode: str | None,
    not_active_reason: str | None,
    partial: bool,
    exit_code: int | None,
    allow_partial: bool,
) -> tuple[str, str]:
    """Return the state and the one sentence that explains it."""
    if active_mode is not None:
        if active_mode == "off":
            # `off` is the stock arm, so announcing it is not engagement.
            if requested_mode == "off":
                return (STOCK_BY_REQUEST, _STOCK_BY_REQUEST_DETAIL)
            return (
                SILENT_STOCK,
                f"the run asked for {requested_mode} and the kit announced the stock arm",
            )
        if active_mode != requested_mode:
            # An accelerated build did run. It is the wrong one, which is a
            # refusal, and reporting it as though nothing engaged would be false.
            return (
                MODE_MISMATCH,
                f"the run asked for {requested_mode} and {active_mode} engaged instead, "
                f"so an accelerated build ran and it is not the one that was asked for",
            )
        if partial:
            return (ACTIVE_PARTIAL, _partial_detail(active_mode, allow_partial, None))
        return (ACTIVE, f"{active_mode} engaged")

    if not_active_reason is not None:
        stock_arm = bool(_STOCK_REASON.match(not_active_reason)) or not_active_mode == "off"
        if stock_arm:
            if requested_mode == "off":
                return (STOCK_BY_REQUEST, _STOCK_BY_REQUEST_DETAIL)
            return (
                SILENT_STOCK,
                f"the run asked for {requested_mode} and the kit reported the stock arm instead",
            )
        if _PARTIAL_REASON.match(not_active_reason) or partial:
            return (
                ACTIVE_PARTIAL,
                _partial_detail(requested_mode, allow_partial, not_active_reason),
            )
        return (NOT_ACTIVE, f"the kit refused to engage: {not_active_reason}")

    if exit_code not in (None, EXIT_OK):
        return (
            NOT_ACTIVE,
            f"the kit printed no mode line and exited {exit_code}: "
            f"{EXIT_MEANINGS.get(exit_code, 'unrecognised exit code')}",
        )
    return (
        NO_ANNOUNCEMENT,
        "the kit printed no ACTIVE and no NOT ACTIVE line, so nothing here states which "
        "build ran and the output cannot be attributed to a mode",
    )


def require_engaged(verdict: Mapping[str, Any]) -> None:
    """Raise unless the run may be attributed to the mode it asked for.

    A stock run that asked for stock passes. Everything else that did not engage
    raises, including the two quiet ones: a kit that served stock under an
    accelerated request, and a kit that announced nothing at all.

    Two refusals are not "nothing ran". A mode mismatch ran an accelerated build
    that was not the one asked for, and a refused partial ran the right mode
    without every optimization. The message says which case it is, because
    "the fast build did not run" would be false for both.
    """
    if verdict.get("accepted"):
        return
    requested = verdict.get("requested_mode")
    if verdict.get("engaged"):
        opening = (
            f"the run may not be attributed to {requested}, although a build did engage"
        )
    else:
        opening = f"the {requested} build did not run"
    raise KitEngagementError(
        f"{opening}: {verdict.get('detail')}. "
        f"Accelerated timings and outputs may not be attributed to {requested}. "
        f"Re-run with the kit's own check for this card and mode before spending again."
    )


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise KitEngagementError(f"cannot read {path}: {error}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    assess = sub.add_parser(
        "assess", help="read a captured kit log and report which build ran"
    )
    assess.add_argument("--mode", required=True, choices=KIT_MODES)
    assess.add_argument("--stdout", type=Path, default=None)
    assess.add_argument("--stderr", type=Path, default=None)
    assess.add_argument("--exit-code", type=int, default=None)
    assess.add_argument("--allow-partial", action="store_true")
    assess.add_argument("--result-path", type=Path, default=None)
    assess.add_argument(
        "--require-engaged",
        action="store_true",
        help="exit 3 unless the run may be attributed to the mode it asked for",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    stdout = _read(arguments.stdout) if arguments.stdout else ""
    stderr = _read(arguments.stderr) if arguments.stderr else ""
    if not stdout and not stderr:
        raise KitEngagementError("assess needs --stdout or --stderr; neither was given")
    verdict = parse_engagement(
        requested_mode=arguments.mode,
        stdout=stdout,
        stderr=stderr,
        exit_code=arguments.exit_code,
        allow_partial=arguments.allow_partial,
    )
    rendered = json.dumps(verdict, indent=2, sort_keys=True)
    if arguments.result_path:
        arguments.result_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.result_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if arguments.require_engaged and not verdict["accepted"]:
        print(f"NOT ATTRIBUTABLE: {verdict['detail']}", file=sys.stderr)
        return EXIT_NOT_ACTIVE
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KitEngagementError as error:
        print(f"kit engagement: {error}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)
