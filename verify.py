#!/usr/bin/env python3
"""Verify the released files and optionally exercise the free planning workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


MANIFEST = "RELEASE-MANIFEST.json"
SKILL = Path("skills/claude-binder-lane")
ROOT_FILES = {"README.md", "LICENSE.txt", "verify.py", MANIFEST, "skills", "assets", ".gitignore"}
BYTECODE_DIRECTORY = "__pycache__"
BYTECODE_SUFFIXES = (".pyc", ".pyo")


def is_bytecode(name: str) -> bool:
    """Report whether a release-relative path is an interpreter bytecode cache.

    Bytecode is reported rather than ignored. A `.pyc` whose header matches the source
    file's size and timestamp is imported in place of that source, and a hash-based
    `.pyc` marked unchecked is imported without any comparison at all. Skipping these
    files would let the code an install actually runs differ from the code this check
    hashed, which is the one guarantee the manifest exists to give.
    """
    return BYTECODE_DIRECTORY in PurePosixPath(name).parts or name.endswith(BYTECODE_SUFFIXES)


def bytecode_remedy(root: Path, cached: list[str]) -> str:
    """Name the cause of a bytecode cache in the release tree, and the remedy."""
    return (
        f"importing the package wrote {len(cached)} bytecode cache file(s) into this "
        f"release, the first being {cached[0]}.\n"
        "The install itself is not damaged, and the source files still match the manifest.\n"
        "A bytecode cache is reported rather than ignored, because a .pyc is imported in "
        "place of the source file the manifest hashes.\n"
        "Delete the caches:\n"
        f"  find {root} \\( -name {BYTECODE_DIRECTORY} -type d -prune -o -name '*.py[co]' \\) "
        "-exec rm -rf {} +\n"
        "Then re-run this verifier, and pass -B to python3 or set PYTHONDONTWRITEBYTECODE=1 "
        "so the next import writes none."
    )


def files_under(root: Path) -> dict[str, Path]:
    files = {}
    for entry in root.iterdir():
        if entry.name == ".git" and entry.is_dir() and not entry.is_symlink():
            continue
        # A root `__pycache__` is collected rather than refused here, so the comparison
        # below can report it as the bytecode it is instead of as a strange directory.
        if entry.name not in ROOT_FILES and entry.name != BYTECODE_DIRECTORY:
            raise ValueError(f"unexpected root entry: {entry.name}")
        for path in [entry, *sorted(entry.rglob("*"))] if entry.is_dir() else [entry]:
            if path.is_symlink():
                raise ValueError(f"symlink in export: {path.relative_to(root)}")
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path
    return files


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path) -> dict:
    actual = files_under(root)
    document = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported release manifest version")
    expected = {}
    for row in document["files"]:
        name = row["path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or path.as_posix() != name or not path.parts or name == MANIFEST):
            raise ValueError("invalid release member path")
        # A manifest that lists a cache would make that cache an accepted member, which
        # is the one way a `.pyc` could pass this check. No release ships one.
        if is_bytecode(name):
            raise ValueError(f"the release manifest lists a bytecode cache: {name}")
        if name in expected:
            raise ValueError(f"duplicate release member: {name}")
        expected[name] = row
    actual.pop(MANIFEST, None)
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        cached = [name for name in extra if is_bytecode(name)]
        unexpected = [name for name in extra if not is_bytecode(name)]
        if cached and not missing and not unexpected:
            raise ValueError(bytecode_remedy(root, cached))
        detail = f"release member mismatch: missing={missing}, unexpected={unexpected}"
        if cached:
            detail = f"{detail}\nThe release also holds bytecode caches. {bytecode_remedy(root, cached)}"
        raise ValueError(detail)
    for name, path in actual.items():
        row = expected[name]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"changed release member: {name}")
    skill_files = sorted(
        (name[len(SKILL.as_posix()) + 1:], sha256(path))
        for name, path in actual.items()
        if name.startswith(SKILL.as_posix() + "/")
        and path.name != ".claude-binder-skill-build"
    )
    digest = hashlib.sha256()
    for name, checksum in skill_files:
        digest.update(f"{checksum}  {name}\n".encode())
    if digest.hexdigest() != document["skill_manifest_sha256"]:
        raise ValueError("skill identity differs from the release manifest")
    marker = (root / SKILL / ".claude-binder-skill-build").read_text()
    if f"manifest-sha256: {digest.hexdigest()}\n" not in marker:
        raise ValueError("skill build marker differs from its files")
    return {"ok": True, "files": len(actual), "skill_manifest_sha256": digest.hexdigest()}


KERNEL_COMMAND = r'''
import json
import socket
import sys
from pathlib import Path

def no_network(*args, **kwargs):
    raise RuntimeError("offline check attempted a network connection")

socket.create_connection = no_network
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.getaddrinfo = no_network
skill = Path(sys.argv[1]).resolve()
kernel = skill / "kernel.py"
namespace = {"__name__": "kernel"}
exec(compile(kernel.read_text(encoding="utf-8"), str(kernel), "exec"), namespace)
version = namespace["binder_version"]()
assert version["source"] == "sibling-files", version
assert Path(version["path"]).resolve() == skill / "claude_binder/__init__.py", version
if len(sys.argv) == 2:
    print(json.dumps(version))
else:
    raise SystemExit(namespace["binder_cli"](sys.argv[2:]))
'''


FIXTURE_RUN_ROOT = "run-data/runs/local-contract-v1"
EXECUTE_TIMEOUT = 1800
# The preflight step alone can use about 80 seconds of CPU, so a busy machine
# needs a wide margin. A fixed 120 seconds failed the smoke check under load.
STEP_TIMEOUT = 900


def _stage_list(value: object, label: str) -> list:
    """Return a list of stage ids, refusing anything that only looks like one.

    A bare string satisfies `len()` and iterates, so `"ab"` would compare as the two
    stages `a` and `b`. A dict does the same over its keys.
    """
    if not isinstance(value, list):
        raise ValueError(f"{label} is {type(value).__name__}, not a list: {value!r}")
    wrong = [item for item in value if not isinstance(item, str)]
    if wrong:
        raise ValueError(f"{label} holds non-string entries: {wrong}")
    return value


def _number(value: object, label: str) -> float:
    """Return a numeric field, refusing a bool, a string or a null standing in for zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a number: {value!r}")
    return float(value)


def _regular_file(path: Path) -> Path:
    """Refuse a symlink, so evidence cannot point outside the run root it belongs to."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} is not a regular file")
    return path


def run_evidence(directory: Path) -> dict:
    """Read what the executed fixture wrote, not the verdict it printed.

    `execute` reports its own success, so a check that reads only that verdict is the
    run marking its own paper. Every figure below comes off the run root instead, and
    the two spend claims a free run has to make are proved rather than assumed.
    """
    run_root = directory / FIXTURE_RUN_ROOT
    plan = json.loads((directory / "run-bundle/run-plan.json").read_text(encoding="utf-8"))
    planned = _stage_list(plan.get("ordered_stage_ids"), "run-plan.json ordered_stage_ids")
    if not planned:
        raise ValueError("the run plan orders no stages, so there is nothing to verify")
    status = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed" or status.get("ok") is not True:
        raise ValueError(f"the fixture run did not complete: {status.get('state')}")
    # `completed_stages` and `skipped_stages` are lists of stage ids, not counts.
    completed = _stage_list(status.get("completed_stages"), "status.json completed_stages")
    # Compare identities, never totals. Counting alone accepted a run that completed a
    # stage the plan never ordered while silently never running one it did, because the
    # two errors cancel in the total. Duplicates are rejected first, so a repeated id
    # cannot pad the count up to the planned length either.
    for label, stages in (("the run plan orders", planned), ("completed_stages repeats", completed)):
        repeated = sorted({stage for stage in stages if stages.count(stage) > 1})
        if repeated:
            raise ValueError(f"{label} the same stage more than once: {repeated}")
    unplanned = sorted(set(completed) - set(planned))
    never_run = sorted(set(planned) - set(completed))
    if unplanned or never_run:
        raise ValueError(
            "the completed stages are not the planned stages: "
            f"{len(completed)} completed against {len(planned)} planned, "
            f"not in the plan {unplanned}, never run {never_run}"
        )
    # An absent key is not evidence of an empty one, so the list has to be there.
    skipped = _stage_list(status.get("skipped_stages"), "status.json skipped_stages")
    if skipped:
        raise ValueError(f"the fixture run skipped stages: {skipped}")
    # `provider_facing_count` is the discriminator, because `count` is computed for every
    # stage including the local fixture adapters and is nonzero on an honest free run.
    # Each counter is still required to be present and numeric, so an omitted or null
    # field cannot read as a verified zero, and `count` is reported rather than asserted.
    calls = status.get("provider_calls")
    if not isinstance(calls, dict):
        raise ValueError(f"status.json records no provider_calls object: {calls!r}")
    for field in ("provider_facing_count", "count", "provider_facing_estimate"):
        if field not in calls:
            raise ValueError(f"status.json provider_calls omits {field}")
        _number(calls[field], f"provider_calls.{field}")
    if _number(calls["provider_facing_count"], "provider_facing_count") != 0:
        raise ValueError(
            f"the free fixture recorded provider-facing calls: {calls['provider_facing_count']!r}"
        )
    # `provider_facing_estimate` is asserted on by nothing, deliberately. It is a ceiling
    # computed from the config rather than a record of contact, and the free fixture's own
    # plan estimates 235 provider calls while its local adapters make none. Requiring it to
    # be zero rejected the honest run. It is required to be present and numeric, and
    # reported, so a reader can see the ceiling the measured zero sits under.
    receipt_dir = run_root / "artifacts" / "receipts"
    receipts = sorted(receipt_dir.glob("*.json"))
    # A non-recursive glob would not see a contradicting receipt written one level down.
    nested = sorted(path for path in receipt_dir.rglob("*.json") if path.parent != receipt_dir)
    if nested:
        raise ValueError(f"receipts are nested below the receipts directory: {nested}")
    for path in receipts:
        _regular_file(path)
    final = [path for path in receipts if not path.name.endswith(".started.json")]
    unfinished = []
    mislabelled = []
    for path in final:
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("ok") is not True:
            unfinished.append(path.name)
        # The file name is not evidence about its contents. A receipt whose body names a
        # different stage is either a copied file or a stage writing another's result, and
        # reading only the name cannot tell either from a real one.
        named = body.get("stage_id")
        if named != path.name[: -len(".json")]:
            mislabelled.append(f"{path.name} carries stage_id {named!r}")
    if unfinished:
        raise ValueError(f"receipts report failure: {unfinished}")
    if mislabelled:
        raise ValueError(f"receipt bodies name a different stage than their file: {mislabelled}")
    if sorted(path.name[: -len(".json")] for path in final) != sorted(completed):
        raise ValueError("the receipts and the completed stage list name different stages")
    sentinels = sorted(
        path.name[: -len(".started.json")]
        for path in receipts
        if path.name.endswith(".started.json")
    )
    if sentinels != sorted(completed):
        raise ValueError(
            "the started sentinels and the completed stage list name different stages"
        )
    # A real spend row carries four amount-bearing fields, and reading two of them let a
    # charge recorded in either of the others pass as free.
    spend_path = run_root / "artifacts" / "spend.jsonl"
    _regular_file(spend_path)
    charged = []
    for line in spend_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for field in (
            "amount",
            "cumulative_amount",
            "cumulative_settled_amount",
            "cumulative_estimated_amount",
        ):
            if field in row and _number(row[field], f"spend {field}") != 0:
                charged.append(f"{row.get('event')}: {field}={row[field]!r}")
    if charged:
        raise ValueError(f"the free fixture recorded spend: {charged}")
    return {
        "planned_stages": len(planned),
        "completed_stages": len(completed),
        "skipped_stages": 0,
        "final_receipts": len(final),
        "started_sentinels": len(receipts) - len(final),
        "provider_facing_calls": 0,
        "stage_invocations_counted": calls["count"],
        "provider_facing_estimate": calls["provider_facing_estimate"],
        "spend_rows_carrying_an_amount": 0,
        "structure_pictures": len(list(run_root.rglob("*.png"))),
    }


def smoke(root: Path, execute: bool = False) -> dict:
    skill = root / SKILL
    data = skill / "claude_binder/data"
    # The smoke uses only fixture planning commands and inherits no credentials.
    environment = {"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"}
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    commands = [
        [],
        ["compose", "--campaign", str(data / "fixtures/local-contract/campaign.json"),
         "--profile", str(data / "templates/profiles/local-contract-test.json"),
         "--out", "composed.json", "--json"],
        ["check", "--config", "composed.json", "--check-paths", "--json"],
        ["materialize", "--config", "composed.json", "--out", "run-bundle",
         "--data-root", "run-data", "--json"],
        ["contract-audit", "--config", "run-bundle/config.resolved.json",
         "--plan", "run-bundle/run-plan.json"],
        ["preflight", "--config", "run-bundle/config.resolved.json",
         "--plan", "run-bundle/run-plan.json"],
    ]
    if execute:
        # Step 6 of the free first run. Every stage is a local fixture adapter, so this
        # starts no provider job, and run_evidence proves that from the receipts after.
        commands.append(
            ["execute", "--plan", "run-bundle/run-plan.json",
             "--run-root", FIXTURE_RUN_ROOT, "--stage", "all", "--json"]
        )
    completed = []
    with tempfile.TemporaryDirectory(prefix="binder-smoke-") as directory:
        for command in commands:
            name = command[0] if command else "sibling-package-loading"
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-S", "-c", KERNEL_COMMAND, str(skill), *command],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=EXECUTE_TIMEOUT if name == "execute" else STEP_TIMEOUT, check=False,
            )
            if result.returncode:
                raise ValueError(f"{name} failed:\n{result.stdout}\n{result.stderr}")
            if command and command[-1] == "--json":
                if json.loads(result.stdout).get("ok") is not True:
                    raise ValueError(f"{name} returned no successful verdict")
            elif command and "ok: True" not in result.stdout:
                raise ValueError(f"{name} returned no successful verdict")
            completed.append(name)
        report = {"ok": True, "checks": completed, "provider_jobs": 0}
        if execute:
            report["run"] = run_evidence(Path(directory))
    return report


def verify_staged(root: Path) -> dict:
    """Check the index that Git will commit, including an alternate test index."""
    rows = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z"], cwd=root,
    ).split(b"\0")
    members = []
    for row in filter(None, rows):
        header, raw_name = row.split(b"\t", 1)
        mode, object_id, stage = header.split()
        name = raw_name.decode("utf-8")
        path = PurePosixPath(name)
        if (mode not in {b"100644", b"100755"} or stage != b"0"
                or path.is_absolute() or ".." in path.parts or "\\" in name
                or path.as_posix() != name or path.parts[0].lower() == ".git"):
            raise ValueError("staged payload contains an unsafe path, link, or unresolved merge")
        members.append((name, object_id.decode("ascii")))
    with tempfile.TemporaryDirectory(prefix="binder-staged-") as directory:
        # Read raw blobs so checkout filters and line-ending conversion cannot
        # replace the staged bytes with a different working representation.
        for name, object_id in members:
            destination = Path(directory) / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.check_output(
                ["git", "cat-file", "blob", object_id], cwd=root,
            ))
        report = verify(Path(directory))
    report["snapshot"] = "git-index"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run the five free planning steps")
    mode.add_argument(
        "--acceptance", action="store_true",
        help="run the five planning steps and then execute the free fixture end to end",
    )
    mode.add_argument("--staged", action="store_true", help="verify the exact Git index before committing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        report = verify_staged(root) if args.staged else verify(root)
        if args.smoke or args.acceptance:
            report["smoke"] = smoke(root, execute=args.acceptance)
            verify(root)
        print(json.dumps(report, indent=2))
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"Check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
