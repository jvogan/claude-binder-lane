#!/usr/bin/env bash
# scripts/smoke-then-scale.sh
#
# Bio-campaign SMOKE_FIRST discipline — a reusable bash library + CLI for the
# smoke-then-scale dispatch pattern. Source this file (or call as a script with
# `smoke_then_scale ...`) from any tool-specific runner before launching a
# multi-design / multi-sample bio-design job (PepGLAD, RFpeptides, BindCraft,
# Boltz, Chai-1, Genie 3, RFdiffusion3, …).
#
# WHY THIS EXISTS
# ---------------
# Two campaigns in May 2026 each lost about 30 minutes to cascading failures
# that a five-minute N=1 smoke would have caught:
#
#   * One arm installed a package, failed silently, never extracted a model
#     checkpoint, crashed on load, and produced 0 PDBs. The stage still
#     reported complete, because the script exited 0.
#   * Another jumped straight to N=50 and hit successive bugs in environment
#     install and output path schema. Each one cost a restart.
#
# Both read the exit code as the result. The fix is structural:
# every dispatch MUST run N=1 first, MUST assert outputs exist + parse, and
# MUST validate output count before declaring any stage complete.
#
# CONTRACT
# --------
# After this script runs to success, the following markers exist in $WORKDIR:
#   STAGE_SMOKE_PASSED   — N=1 produced valid output and parsed cleanly
#   STAGE_SCALE_PASSED   — N=$TARGET_N produced at least $TARGET_N outputs
#                          AND each downstream-assertion succeeded
# On failure, exactly one of:
#   STAGE_SMOKE_FAILED   — smoke command, glob, or parse gate failed
#   STAGE_SCALE_FAILED   — scale stage failed an output count or assertion
# is written, with the failure reason in $WORKDIR/dispatch.log and a final
# line in $WORKDIR/smoke-then-scale.events.jsonl.
#
# A `STAGE_COMPLETE` marker is NEVER written by this library — that token is
# reserved for the orchestrator after it has independently verified outputs.
#
# USAGE (sourced)
# ---------------
#   source /workspace/repo/scripts/smoke-then-scale.sh
#   export WORKDIR=/workspace/runs/my-campaign
#   smoke_then_scale \
#     --workdir   "$WORKDIR" \
#     --tool      pepglad \
#     --target-n  50 \
#     --smoke-cmd 'python -m api.run --mode codesign --pdb $PDB \
#                  --pocket $POCKET --out_dir $WORKDIR/smoke \
#                  --length_min 10 --length_max 15 --n_samples 1' \
#     --smoke-out-glob "$WORKDIR/smoke/*.pdb" \
#     --smoke-out-min 1 \
#     --smoke-parse-cmd 'python3 -c "import sys,pathlib; \
#         lines=pathlib.Path(\"$WORKDIR/smoke/summary.jsonl\").read_text().splitlines(); \
#         import json; [json.loads(l) for l in lines]; print(\"parse OK\")"' \
#     --scale-cmd 'python -m api.run --mode codesign --pdb $PDB \
#                  --pocket $POCKET --out_dir $WORKDIR/scale \
#                  --length_min 10 --length_max 15 --n_samples 50' \
#     --scale-out-glob "$WORKDIR/scale/*.pdb" \
#     --scale-out-min 50
#
# USAGE (as CLI)
# --------------
#   bash scripts/smoke-then-scale.sh --workdir ... --tool ... [same flags]
#
# DESIGN NOTES
# ------------
# * Tool-agnostic: every command and assertion is passed in. We do not bake in
#   PepGLAD/Boltz/Chai-specific path knowledge.
# * Fail-loud: every gate is a hard exit on failure. No "continue on warning"
#   behavior. The whole point is that silent failures get caught.
# * Markers are touched at the WORKDIR root with names ALL_CAPS_UNDERSCORED
#   so an orchestrator can `find $WORKDIR -maxdepth 1 -name STAGE_*` grep them.
# * Each transition also appends a single JSON line to smoke-then-scale.events.jsonl
#   for machine-readable progress.
# * The wrapper does NOT swallow the inner command's stderr. Inner stderr lands
#   in $WORKDIR/dispatch.log and is tee'd to current stderr.

set -euo pipefail

# Resolve absolute path to this library so callers can find the canonical
# location regardless of CWD.
_SMOKE_THEN_SCALE_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke-then-scale.sh"
_STS_PYTHON="${CLAUDE_BINDER_PYTHON_EXECUTABLE:-python3}"

# ---------------------------------------------------------------------------
# Internal helpers — prefixed `_sts_` so they do not collide with the
# `sf_stage_*` helpers a calling harness may already define.
# ---------------------------------------------------------------------------

_sts_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# _sts_log <workdir> <message...>
#   Appends the message to $workdir/dispatch.log AND emits to stderr.
_sts_log() {
  local workdir="$1"
  shift
  local ts
  ts="$(_sts_now)"
  local line="[${ts}] $*"
  mkdir -p "$workdir"
  printf '%s\n' "$line" | tee -a "$workdir/dispatch.log" >&2
}

# _sts_event <workdir> <stage> <status> <message>
#   Appends a JSON line to $workdir/smoke-then-scale.events.jsonl.
_sts_event() {
  local workdir="$1" stage="$2" status="$3" message="$4"
  mkdir -p "$workdir"
  "$_STS_PYTHON" - "$workdir/smoke-then-scale.events.jsonl" "$stage" "$status" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, stage, status, message = sys.argv[1:5]
event = {
    "schema_version": 1,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": stage,
    "status": status,
    "message": message,
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(event, sort_keys=True) + "\n")
PY
}

# _sts_touch_marker <workdir> <marker>
#   Atomically writes a stage marker at $workdir/<marker>.
#   Refuses to overwrite an existing marker (defensive — markers should be
#   one-shot per run).
_sts_touch_marker() {
  local workdir="$1" marker="$2"
  mkdir -p "$workdir"
  if [[ -e "$workdir/$marker" ]]; then
    _sts_log "$workdir" "WARN: marker $marker already exists; not overwriting"
    return 0
  fi
  printf '%s\n' "$(_sts_now)" > "$workdir/$marker"
}

# _sts_glob_count <pattern>
#   echoes the number of regular files matching the glob. Returns 0 (not error)
#   on no-match, so callers can compare numerically.
_sts_glob_count() {
  local pattern="$1"
  "$_STS_PYTHON" - "$pattern" <<'PY'
import glob
import os
import sys

print(sum(os.path.isfile(path) for path in glob.glob(sys.argv[1])))
PY
}

# _sts_run_inner <workdir> <label> <cmd>
#   Runs <cmd> in a subshell and appends stderr and stdout to $workdir/dispatch.log.
#   Returns the inner command's exit code.
_sts_run_inner() {
  local workdir="$1" label="$2" cmd="$3"
  _sts_log "$workdir" "RUN [$label] $cmd"
  # Run the command without process substitution so the helper works in
  # sandboxes that do not permit opening /dev/fd descriptors.
  set +e
  ( eval "$cmd" ) >> "$workdir/dispatch.log" 2>&1
  local rc=$?
  set -e
  _sts_log "$workdir" "EXIT [$label] rc=$rc"
  return "$rc"
}

# _sts_die <workdir> <stage> <marker> <message>
#   Records the failure, writes the marker, and exits 1.
_sts_die() {
  local workdir="$1" stage="$2" marker="$3" message="$4"
  _sts_log "$workdir" "FAIL [$stage] $message"
  _sts_event "$workdir" "$stage" "failed" "$message"
  _sts_touch_marker "$workdir" "$marker"
  exit 1
}

# ---------------------------------------------------------------------------
# Public API: smoke_then_scale
# ---------------------------------------------------------------------------
#
# Required flags:
#   --workdir <path>            Absolute dir where markers + logs go.
#   --tool <name>               Free-form tool tag (pepglad, rfpeptides, boltz, …).
#   --target-n <int>            The N you want at scale.
#   --smoke-cmd <shell-cmd>     One-line shell command for N=1 smoke pass.
#   --scale-cmd <shell-cmd>     One-line shell command for the scale pass.
#
# Either-or (at least one must be present so the smoke assertion is meaningful):
#   --smoke-out-glob <glob>     Glob whose match count is checked vs --smoke-out-min.
#   --smoke-parse-cmd <cmd>     Shell command that exits 0 iff smoke output parses.
#
# Optional:
#   --smoke-out-min <int>       Min matches required for smoke (default 1).
#   --scale-out-glob <glob>     Glob whose match count is checked vs --scale-out-min.
#   --scale-out-min <int>       Min matches required for scale (default = target-n).
#   --scale-parse-cmd <cmd>     Shell command that exits 0 iff scale output parses.
#   --skip-scale                If set, smoke-only (e.g. for explicit dry-runs).
#   --allow-rerun               If set, accept pre-existing markers and skip the
#                               corresponding stage. Default: refuse to clobber.
#
# Exit codes:
#   0  = both stages passed (or smoke passed and --skip-scale).
#   1  = smoke or scale failed; appropriate STAGE_*_FAILED marker is touched.
smoke_then_scale() {
  local workdir="" tool="" target_n=""
  local smoke_cmd="" smoke_out_glob="" smoke_out_min="1" smoke_parse_cmd=""
  local scale_cmd="" scale_out_glob="" scale_out_min="" scale_parse_cmd=""
  local skip_scale=0 allow_rerun=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --workdir) workdir="$2"; shift 2 ;;
      --tool) tool="$2"; shift 2 ;;
      --target-n) target_n="$2"; shift 2 ;;
      --smoke-cmd) smoke_cmd="$2"; shift 2 ;;
      --smoke-out-glob) smoke_out_glob="$2"; shift 2 ;;
      --smoke-out-min) smoke_out_min="$2"; shift 2 ;;
      --smoke-parse-cmd) smoke_parse_cmd="$2"; shift 2 ;;
      --scale-cmd) scale_cmd="$2"; shift 2 ;;
      --scale-out-glob) scale_out_glob="$2"; shift 2 ;;
      --scale-out-min) scale_out_min="$2"; shift 2 ;;
      --scale-parse-cmd) scale_parse_cmd="$2"; shift 2 ;;
      --skip-scale) skip_scale=1; shift ;;
      --allow-rerun) allow_rerun=1; shift ;;
      -h|--help)
        sed -n '4,90p' "$_SMOKE_THEN_SCALE_LIB"
        return 0
        ;;
      *)
        echo "smoke_then_scale: unknown flag: $1" >&2
        return 2
        ;;
    esac
  done

  # --- Required-flag validation ---
  local missing=()
  [[ -n "$workdir"   ]] || missing+=("--workdir")
  [[ -n "$tool"      ]] || missing+=("--tool")
  [[ -n "$target_n"  ]] || missing+=("--target-n")
  [[ -n "$smoke_cmd" ]] || missing+=("--smoke-cmd")
  [[ -n "$scale_cmd" || "$skip_scale" -eq 1 ]] || missing+=("--scale-cmd (or --skip-scale)")
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "smoke_then_scale: missing required flag(s): ${missing[*]}" >&2
    return 2
  fi
  if [[ -z "$smoke_out_glob" && -z "$smoke_parse_cmd" ]]; then
    echo "smoke_then_scale: must provide at least one of --smoke-out-glob or --smoke-parse-cmd" >&2
    return 2
  fi

  # Default scale_out_min to target_n if not provided.
  [[ -n "$scale_out_min" ]] || scale_out_min="$target_n"

  mkdir -p "$workdir"
  _sts_log "$workdir" "smoke_then_scale starting tool=$tool target_n=$target_n workdir=$workdir"
  _sts_event "$workdir" "init" "started" "tool=$tool target_n=$target_n"

  # --- Pre-flight: refuse to clobber markers unless --allow-rerun ---
  local clobber_markers=()
  for m in STAGE_SMOKE_PASSED STAGE_SMOKE_FAILED STAGE_SCALE_PASSED STAGE_SCALE_FAILED; do
    [[ -e "$workdir/$m" ]] && clobber_markers+=("$m")
  done
  if [[ ${#clobber_markers[@]} -gt 0 && "$allow_rerun" -eq 0 ]]; then
    _sts_die "$workdir" "init" "STAGE_SMOKE_FAILED" \
      "pre-existing markers in workdir: ${clobber_markers[*]}; pass --allow-rerun or clear them"
  fi

  # ===========================================================================
  # SMOKE STAGE
  # ===========================================================================
  if [[ "$allow_rerun" -eq 1 && -e "$workdir/STAGE_SMOKE_PASSED" ]]; then
    _sts_log "$workdir" "SKIP smoke (already passed, --allow-rerun)"
    _sts_event "$workdir" "smoke" "skipped" "STAGE_SMOKE_PASSED already exists"
  else
    # Wipe stale FAILED marker if re-running with --allow-rerun
    rm -f "$workdir/STAGE_SMOKE_FAILED"

    _sts_event "$workdir" "smoke" "started" "n=1"
    if ! _sts_run_inner "$workdir" "smoke" "$smoke_cmd"; then
      _sts_die "$workdir" "smoke" "STAGE_SMOKE_FAILED" "smoke command exited non-zero"
    fi

    # Parse assertion. This runs before the glob because the glob names the stage's
    # declared artifact, and for several adapters the parse command is what writes it.
    # The parse command fails on zero inputs, so the anti-cascade property survives:
    # a tool that exits 0 having produced nothing still stops here.
    if [[ -n "$smoke_parse_cmd" ]]; then
      if ! _sts_run_inner "$workdir" "smoke-parse" "$smoke_parse_cmd"; then
        _sts_die "$workdir" "smoke" "STAGE_SMOKE_FAILED" "smoke parse assertion failed"
      fi
    fi

    # Output-glob assertion
    if [[ -n "$smoke_out_glob" ]]; then
      local n
      n=$(_sts_glob_count "$smoke_out_glob")
      _sts_log "$workdir" "smoke glob '$smoke_out_glob' matched $n files (require >= $smoke_out_min)"
      if [[ "$n" -lt "$smoke_out_min" ]]; then
        _sts_die "$workdir" "smoke" "STAGE_SMOKE_FAILED" \
          "smoke output glob matched $n < $smoke_out_min ($smoke_out_glob)"
      fi
    fi

    _sts_touch_marker "$workdir" "STAGE_SMOKE_PASSED"
    _sts_event "$workdir" "smoke" "passed" "smoke gates ok"
    _sts_log "$workdir" "OK  STAGE_SMOKE_PASSED"
  fi

  if [[ "$skip_scale" -eq 1 ]]; then
    _sts_log "$workdir" "skip_scale=1, exiting after smoke"
    _sts_event "$workdir" "scale" "skipped" "--skip-scale"
    return 0
  fi

  # ===========================================================================
  # SCALE STAGE
  # ===========================================================================
  rm -f "$workdir/STAGE_SCALE_FAILED"
  _sts_event "$workdir" "scale" "started" "n=$target_n"
  if ! _sts_run_inner "$workdir" "scale" "$scale_cmd"; then
    _sts_die "$workdir" "scale" "STAGE_SCALE_FAILED" "scale command exited non-zero"
  fi

  # Parse assertion, before the glob, for the reason given in the smoke block above.
  if [[ -n "$scale_parse_cmd" ]]; then
    if ! _sts_run_inner "$workdir" "scale-parse" "$scale_parse_cmd"; then
      _sts_die "$workdir" "scale" "STAGE_SCALE_FAILED" "scale parse assertion failed"
    fi
  fi

  # Output-glob assertion (the v3 anti-cascade fix: count outputs, do not trust exit code)
  if [[ -n "$scale_out_glob" ]]; then
    local n
    n=$(_sts_glob_count "$scale_out_glob")
    _sts_log "$workdir" "scale glob '$scale_out_glob' matched $n files (require >= $scale_out_min)"
    if [[ "$n" -lt "$scale_out_min" ]]; then
      _sts_die "$workdir" "scale" "STAGE_SCALE_FAILED" \
        "scale output glob matched $n < $scale_out_min ($scale_out_glob)"
    fi
  else
    _sts_log "$workdir" "WARN: no --scale-out-glob provided; cannot count outputs (less safe)"
  fi

  _sts_touch_marker "$workdir" "STAGE_SCALE_PASSED"
  _sts_event "$workdir" "scale" "passed" "scale gates ok"
  _sts_log "$workdir" "OK  STAGE_SCALE_PASSED"
  return 0
}

# ---------------------------------------------------------------------------
# CLI dispatch when not sourced.
# ---------------------------------------------------------------------------
# BASH_SOURCE[0] == $0 when executed directly. When sourced, $0 is the parent shell.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  smoke_then_scale "$@"
fi
