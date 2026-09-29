#!/usr/bin/env python3
"""Bounded small-campaign worker and report tools for Claude Science.

This program never submits a cloud job. ``run-design`` and ``run-rescore`` are
worker entry points: the caller must stage their inputs and a plan-bound,
operator-origin approval record in the approved provider job. The other
commands are local. A run is not complete until provider receipts are joined
to every worker receipt and the score/report gate passes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile


AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")
ARMS = ("esmfold2-kit", "esmfold2-platform", "boltz2-kit")
DEFAULT_RANKING_RULE = "mean_of_arm_control_normalized_mean_ipsae"
RANKING_RULES = (DEFAULT_RANKING_RULE, "mean_of_arm_best_ipsae")
KITS = {"esmfold2-kit": "esmfold2", "boltz2-kit": "boltz2"}
WORKER_MANIFEST = "small-campaign-worker-manifest.json"


def _json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _worker_sources() -> dict[str, Path]:
    """Select the installed worker code without local caches or test fixtures."""
    skill_root = Path(__file__).resolve().parents[1]
    package = skill_root / "claude_binder"
    if not package.is_dir():
        package = skill_root.parents[1] / "src" / "claude_binder"
    _require(package.is_dir(), "installed skill has no sibling claude_binder package")
    files = {
        f"scripts/{name}": skill_root / "scripts" / name
        for name in ("small_campaign.py", "small_campaign_esmfold.py")
    }
    for path in package.rglob("*"):
        if not path.is_file() or any(part in {"tests", "__pycache__"} for part in path.parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name.startswith("."):
            continue
        files[str(Path("claude_binder") / path.relative_to(package))] = path
    _require(all(path.is_file() for path in files.values()), "worker script is missing from installed skill")
    return files


def bundle_worker(args):
    """Create one portable worker upload for the selected provider job."""
    files = _worker_sources()
    manifest = {
        "schema": "small-campaign-worker-bundle-v1",
        "files": {name: _file_digest(path) for name, path in sorted(files.items())},
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr(WORKER_MANIFEST, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"bundle": str(output.resolve()), "sha256": _file_digest(output), "files": len(files)}, sort_keys=True))


def verify_worker(args):
    root = Path(args.root).resolve()
    _require(bool(args.archive) == bool(args.expected_sha256), "verify-worker needs both archive and expected SHA-256")
    if args.archive:
        _require(re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256) is not None, "expected worker archive SHA-256 must be lowercase hex")
        archive = Path(args.archive)
        _require(archive.is_file() and _file_digest(archive) == args.expected_sha256, "staged worker archive digest differs from the local bundle")
    _require((root / "scripts" / "small_campaign.py").resolve() == Path(__file__).resolve(), "worker command is not running from the staged bundle")
    manifest = _json(root / WORKER_MANIFEST)
    _require(manifest.get("schema") == "small-campaign-worker-bundle-v1", "worker bundle manifest schema differs")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "worker bundle manifest has no files")
    for name, digest in files.items():
        path = root / name
        _require(path.is_file() and _file_digest(path) == digest, f"staged worker file missing or changed: {name}")
    _require("scripts/small_campaign_esmfold.py" in files and "claude_binder/__init__.py" in files, "worker bundle lacks runtime components")
    print(json.dumps({"ok": True, "files": len(files)}, sort_keys=True))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require(ok: bool, message: str):
    if not ok:
        raise ValueError(message)


def _positive(value, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be numeric")
    n = float(value)
    _require(math.isfinite(n) and n >= 0, f"{name} must be finite and nonnegative")
    return n


def _validate_settings(settings: dict):
    _require(isinstance(settings, dict), "settings must be an object")
    _require(IDENT.fullmatch(str(settings.get("campaign_id", ""))) is not None, "campaign_id must be a short identifier")
    _require(settings.get("provider") in ("modal", "runpod", "lambda", "local"), "provider must be explicit")
    _require(_positive(settings.get("maximum_spend_usd"), "maximum_spend_usd") > 0, "maximum_spend_usd must be positive")
    _require(_positive(settings.get("maximum_job_estimate_usd"), "maximum_job_estimate_usd") > 0, "maximum_job_estimate_usd must be positive")
    _require(settings["maximum_job_estimate_usd"] <= settings["maximum_spend_usd"], "per-job estimate ceiling exceeds campaign ceiling")
    _require(isinstance(settings.get("hardware"), str) and settings["hardware"].strip(), "hardware must name the planned GPU")
    _require(isinstance(settings.get("egress_domains"), list) and all(isinstance(x, str) and x for x in settings["egress_domains"]), "egress_domains must be a list of hosts")
    destination = Path(str(settings.get("data_destination", "")))
    _require(destination.is_absolute() and ".." not in destination.parts, "data_destination must be an absolute output path")
    target = settings.get("target", {})
    _require(AA.fullmatch(str(target.get("sequence", ""))) is not None, "target.sequence must be an amino-acid sequence")
    _require(target.get("chain") == "A", "small-campaign target chain must be A")
    for key in ("structure_path", "msa_path"):
        _require(Path(str(target.get(key, ""))).is_file(), f"target.{key} must be an existing file")
    design = settings.get("design", {})
    _require(isinstance(design.get("enabled"), bool), "design.enabled must be boolean")
    if design["enabled"]:
        _require(Path(str(design.get("bindcraft_settings_path", ""))).is_file(), "design.bindcraft_settings_path must exist")
        seeds = design.get("seeds")
        _require(isinstance(seeds, list) and seeds and all(type(s) is int and s >= 0 for s in seeds) and len(set(seeds)) == len(seeds), "design.seeds must be distinct nonnegative integers")
        _require(type(design.get("maximum_candidates")) is int and 1 <= design["maximum_candidates"] <= 60, "design.maximum_candidates must be 1..60")
        pattern = design.get("accepted_glob")
        _require(isinstance(pattern, str) and pattern and "__REQUIRED" not in pattern and not Path(pattern).is_absolute() and ".." not in Path(pattern).parts, "design.accepted_glob must be a relative output pattern")
    controls = settings.get("controls", {})
    _require(IDENT.fullmatch(str(controls.get("positive_id", ""))) is not None, "controls.positive_id is required")
    negatives = controls.get("negative_ids")
    _require(isinstance(negatives, list) and negatives and all(IDENT.fullmatch(str(n)) for n in negatives), "controls.negative_ids is required")
    _require(controls["positive_id"] not in negatives and len(set(negatives)) == len(negatives), "control IDs must be distinct")
    rescore = settings.get("rescore", {})
    arms = rescore.get("predictors")
    _require(isinstance(arms, list) and len(arms) == 2 and len(set(arms)) == 2 and all(a in ARMS for a in arms), "rescore.predictors must name two distinct supported arms")
    _require(not set(arms) == {"esmfold2-kit", "esmfold2-platform"}, "two ESMFold2 runtimes are one model lineage; add another predictor")
    seeds = rescore.get("seeds")
    _require(isinstance(seeds, list) and seeds and all(type(seed) is int and seed >= 0 for seed in seeds)
             and len(set(seeds)) == len(seeds), "rescore.seeds must be distinct nonnegative integers")
    _require(rescore.get("ranking_rule", DEFAULT_RANKING_RULE) in RANKING_RULES,
             f"rescore.ranking_rule must be one of {', '.join(RANKING_RULES)}")
    for arm in arms:
        gate = rescore.get("gates", {}).get(arm, {})
        for key in ("candidate_multiplier", "minimum_positive_margin", "maximum_pose_rmsd"):
            _positive(gate.get(key), f"rescore.gates.{arm}.{key}")
        _require(gate["candidate_multiplier"] > 0, "candidate_multiplier must be positive")
        _require(gate.get("negative_reference") in ("largest_negative_best_seed", "largest_negative_mean"), "negative_reference must be explicit")
        _require(isinstance(gate.get("rule_context"), str) and gate["rule_context"].strip(), "rule_context must name the calibration context")
    _require(rescore.get("boltz2_mode") in ("off", "exact", "fast", "big"), "boltz2_mode is required")
    _require(rescore.get("esmfold2_mode") in ("off", "exact", "fast", "big"), "esmfold2_mode is required")
    if "esmfold2-platform" in arms:
        _require(isinstance(rescore.get("esmfold2_revision"), str) and re.fullmatch(r"[0-9a-f]{40}", rescore["esmfold2_revision"]) is not None, "esmfold2-platform requires a 40-hex model revision")


def _validate_jobs(jobs: list, settings: dict):
    _require(isinstance(jobs, list) and jobs, "jobs must be a nonempty array")
    seen = set()
    for row in jobs:
        cid = row.get("name")
        _require(isinstance(cid, str) and IDENT.fullmatch(cid) is not None and cid not in seen, "job names must be distinct safe identifiers")
        seen.add(cid)
        _require(row.get("target") == settings["target"]["sequence"], f"{cid}: target sequence differs from plan")
        _require(AA.fullmatch(str(row.get("binder", ""))) is not None, f"{cid}: binder sequence invalid")
    controls = settings["controls"]
    _require(controls["positive_id"] in seen, "positive control missing from jobs")
    _require(set(controls["negative_ids"]) <= seen, "negative control missing from jobs")
    design_count = len(seen - {controls["positive_id"], *controls["negative_ids"]})
    _require(design_count <= settings["design"].get("maximum_candidates", 100), "candidate count exceeds plan")


def plan(args):
    settings = _json(args.settings)
    _validate_settings(settings)
    jobs = _json(args.jobs) if args.jobs else None
    if jobs is not None:
        _validate_jobs(jobs, settings)
    sources = {}
    for name, path in (("target_structure", settings["target"]["structure_path"]), ("target_msa", settings["target"]["msa_path"])):
        sources[name] = _file_digest(Path(path))
    if settings["design"]["enabled"]:
        sources["bindcraft_settings"] = _file_digest(Path(settings["design"]["bindcraft_settings_path"]))
    body = {"schema": "small-campaign-plan-v1", "created_at_utc": _utc(), "settings": settings, "jobs": jobs, "source_sha256": sources}
    if jobs is not None:
        arms = settings["rescore"]["predictors"]
        seeds = settings["rescore"]["seeds"]
        body["rescore_seed_count"] = len(seeds)
        body["rescore_seed_ids"] = seeds
        body["planned_rescore_jobs"] = len(arms) * len(seeds)
        body["planned_prediction_calls"] = len(jobs) * len(arms) * len(seeds)
    body["plan_sha256"] = _digest(body)
    _write(args.out, body)
    print(body["plan_sha256"])


def make_roster(args):
    """Turn BindCraft's ranked table and explicit controls into a pinned roster."""
    settings = _json(args.settings)
    _validate_settings(settings)
    _require(1 <= args.limit <= settings["design"]["maximum_candidates"], "roster limit exceeds design ceiling")
    _require(args.design_target_chain != args.design_binder_chain, "design target and binder chains must differ")
    table = Path(args.ranked_csv)
    _require(table.is_file(), "Ranked.csv or !_Ranked.csv is absent")
    controls = _json(args.controls)
    _require(isinstance(controls, list), "controls must be an array of name/target/binder rows")
    rows = []
    with table.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require({"design", "Binder_Sequence", "rank"} <= set(reader.fieldnames or []), "ranked table needs design, Binder_Sequence, rank")
        for row in reader:
            name = row["design"]
            _require(IDENT.fullmatch(name) is not None, f"unsafe design name {name!r}")
            if args.target_state_suffix:
                _require(args.target_state_suffix.startswith("_") and IDENT.fullmatch(args.target_state_suffix[1:]) is not None, "target-state-suffix must be a safe _state suffix")
                pose = table.parent / f"{name}{args.target_state_suffix}.cif"
                _require(pose.is_file() and not pose.name.endswith("_monomer.cif"), f"named accepted complex absent: {pose}")
            else:
                poses = sorted(path for path in [table.parent / f"{name}.cif", *table.parent.glob(f"{name}_*.cif")]
                               if path.is_file() and not path.name.endswith("_monomer.cif"))
                _require(len(poses) == 1, f"{name}: expected one accepted complex, found {[p.name for p in poses]}; use --target-state-suffix")
                pose = poses[0]
            binder = row["Binder_Sequence"]
            _require(AA.fullmatch(binder) is not None, f"{name}: binder sequence invalid")
            rows.append({"name": name, "target": settings["target"]["sequence"], "binder": binder,
                         "kind": "design", "design_pose_path": str(pose.resolve()),
                         "design_pose_sha256": _file_digest(pose),
                         "design_target_chain": args.design_target_chain,
                         "design_binder_chain": args.design_binder_chain,
                         "bc2_rank": int(row["rank"])})
    rows.sort(key=lambda row: (row["bc2_rank"], row["name"]))
    rows = rows[:args.limit]
    _require(bool(rows), "ranked table has no accepted design")
    for control in controls:
        _require(control.get("kind") in ("positive_control", "negative_control"), "each control needs explicit positive_control or negative_control kind")
        rows.append(control)
    _validate_jobs(rows, settings)
    _write(args.out, rows)


def index_artifacts(args):
    """Resolve one arm/seed's CIF and PAE files by roster ID, refusing ambiguity."""
    plan_doc = _json(args.plan)
    jobs = plan_doc.get("jobs")
    _require(isinstance(jobs, list) and jobs, "index-artifacts needs a roster-bound plan")
    _require(args.arm in plan_doc["settings"]["rescore"]["predictors"] and args.seed in plan_doc["settings"]["rescore"]["seeds"], "arm or seed outside plan")
    _require(args.target_chain != args.binder_chain and args.reference_target_chain != args.reference_binder_chain, "target and binder chain IDs must differ")
    root = Path(args.prediction_root).resolve()
    _require(root.is_dir(), "prediction root absent")

    def matches(path: Path, name: str) -> bool:
        token = re.compile(r"(?:^|[_-])" + re.escape(name) + r"(?:$|[_.-])")
        return any(token.search(part) is not None for part in path.relative_to(root).parts)

    manifest = []
    for job in jobs:
        name = job["name"]
        cifs = [p for p in root.rglob("*.cif") if matches(p, name)]
        paes = [p for p in root.rglob("*.npz") if matches(p, name) and ("pae" in p.name.lower() or args.arm == "esmfold2-platform")]
        _require(len(cifs) == 1, f"{name}: expected one CIF under {root}, found {len(cifs)}: {[str(p) for p in cifs[:8]]}")
        _require(len(paes) == 1, f"{name}: expected one PAE NPZ under {root}, found {len(paes)}: {[str(p) for p in paes[:8]]}")
        pose = job.get("design_pose_path")
        if job.get("kind") == "design":
            _require(isinstance(pose, str) and Path(pose).is_file() and job.get("design_pose_sha256") == _file_digest(Path(pose)), f"{name}: design pose missing or changed")
        manifest.append({"candidate_id": name, "arm": args.arm, "seed": args.seed, "job_ref": args.job_ref,
                         "complex_path": str(cifs[0]), "pae_npz_path": str(paes[0]),
                         "target_chain": args.target_chain, "binder_chain": args.binder_chain,
                         "reference_target_chain": args.reference_target_chain,
                         "reference_binder_chain": args.reference_binder_chain,
                         "pae_orientation": args.pae_orientation,
                         "design_pose_path": pose})
    _write(args.out, manifest)


def _approved(plan_path: str, approval_path: str):
    plan_doc = _json(plan_path)
    digest = plan_doc.get("plan_sha256")
    _require(digest == _digest({k: v for k, v in plan_doc.items() if k != "plan_sha256"}), "plan digest mismatch")
    approval = _json(approval_path)
    _require(approval.get("plan_sha256") == digest, "approval does not bind this plan")
    _require(approval.get("provider") == plan_doc["settings"]["provider"], "approval provider mismatch")
    _require(_positive(approval.get("maximum_spend_usd"), "approval.maximum_spend_usd") <= plan_doc["settings"]["maximum_spend_usd"], "approval cap exceeds plan")
    _require(_positive(approval.get("maximum_job_estimate_usd"), "approval.maximum_job_estimate_usd") <= plan_doc["settings"]["maximum_job_estimate_usd"], "approval per-job ceiling exceeds plan")
    for key in ("hardware", "egress_domains", "data_destination"):
        _require(approval.get(key) == plan_doc["settings"][key], f"approval {key} differs from plan")
    for key in ("approved_by", "approval_ref", "approved_at_utc"):
        _require(isinstance(approval.get(key), str) and approval[key].strip(), f"approval.{key} is required")
    _require(isinstance(approval.get("campaign_authorization_id"), str) and IDENT.fullmatch(approval["campaign_authorization_id"]) is not None, "approval.campaign_authorization_id is required")
    _require(approval.get("decision") == "approved", "approval decision must be approved")
    settings = plan_doc["settings"]
    sources = plan_doc["source_sha256"]
    for name, path in (("target_structure", settings["target"]["structure_path"]), ("target_msa", settings["target"]["msa_path"])):
        _require(Path(path).is_file() and _file_digest(Path(path)) == sources[name], f"planned input changed or absent: {name}")
    if settings["design"]["enabled"]:
        path = Path(settings["design"]["bindcraft_settings_path"])
        _require(path.is_file() and _file_digest(path) == sources["bindcraft_settings"], "planned BindCraft settings changed or absent")
    return plan_doc, approval


def admit(args):
    """Reserve an estimate under a locked per-campaign ledger before submission.

    The provider controller must call this before creating a paid job. This
    command makes a reservation ticket; it never submits the job itself.
    """
    plan_doc, approval = _approved(args.plan, args.approval)
    _require(IDENT.fullmatch(args.job_ref) is not None, "job reference is required")
    estimate = _positive(args.estimate_usd, "estimate_usd")
    _require(estimate > 0 and estimate <= approval["maximum_job_estimate_usd"], "estimate exceeds approved per-job ceiling")
    _require(args.stage in ("design", "rescore"), "stage must be design or rescore")
    if args.stage == "design":
        _require(plan_doc["jobs"] is None, "design admission needs a design-only plan")
        _require(args.seed in plan_doc["settings"]["design"]["seeds"], "seed outside design plan")
        _require(args.arm is None, "design admission has no predictor arm")
    else:
        _require(plan_doc["jobs"] is not None, "rescore admission needs a roster-bound plan")
        _require(args.arm in plan_doc["settings"]["rescore"]["predictors"], "arm outside rescore plan")
        _require(args.seed in plan_doc["settings"]["rescore"]["seeds"], "seed outside rescore plan")
    ledger = Path(args.ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        entries = [json.loads(line) for line in stream if line.strip()]
        previous_hash = None
        for entry in entries:
            _require(entry.get("previous_hash") == previous_hash, "admission ledger hash chain broken")
            _require(entry.get("record_hash") == _digest({k: v for k, v in entry.items() if k != "record_hash"}), "admission ledger record changed")
            _require(entry.get("campaign_authorization_id") == approval["campaign_authorization_id"] and entry.get("campaign_id") == plan_doc["settings"]["campaign_id"], "admission ledger belongs to another campaign authorization")
            _require(_positive(entry.get("campaign_ceiling_usd"), "ledger campaign ceiling") >= approval["maximum_spend_usd"], "later phase approval increased the original campaign ceiling")
            previous_hash = entry["record_hash"]
        _require(args.job_ref not in {entry["job_ref"] for entry in entries}, "job reference already admitted")
        reserved = sum(float(entry["estimated_cost_usd"]) for entry in entries)
        _require(reserved + estimate <= approval["maximum_spend_usd"], "remaining approved campaign ceiling cannot admit this estimate")
        ticket = {"schema": "small-campaign-admission-v1", "campaign_id": plan_doc["settings"]["campaign_id"], "campaign_authorization_id": approval["campaign_authorization_id"], "campaign_ceiling_usd": approval["maximum_spend_usd"], "plan_sha256": plan_doc["plan_sha256"], "approval_ref": approval["approval_ref"], "job_ref": args.job_ref, "stage": args.stage, "arm": args.arm, "seed": args.seed, "estimated_cost_usd": estimate, "reserved_total_usd": reserved + estimate, "provider": approval["provider"], "hardware": approval["hardware"], "egress_domains": approval["egress_domains"], "data_destination": approval["data_destination"], "admitted_at_utc": _utc(), "previous_hash": previous_hash}
        ticket["record_hash"] = _digest(ticket)
        stream.seek(0, 2)
        stream.write(json.dumps(ticket, sort_keys=True) + "\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    _write(args.out, ticket)
    print(ticket["record_hash"])


def _admitted(ticket_path: str, plan_doc: dict, approval: dict, *, job_ref: str, stage: str, arm: str | None, seed: int):
    ticket = _json(ticket_path)
    _require(ticket.get("record_hash") == _digest({k: v for k, v in ticket.items() if k != "record_hash"}), "admission ticket digest mismatch")
    for key, value in (("campaign_id", plan_doc["settings"]["campaign_id"]), ("campaign_authorization_id", approval["campaign_authorization_id"]), ("plan_sha256", plan_doc["plan_sha256"]), ("approval_ref", approval["approval_ref"]), ("job_ref", job_ref), ("stage", stage), ("arm", arm), ("seed", seed), ("provider", approval["provider"]), ("hardware", approval["hardware"]), ("egress_domains", approval["egress_domains"]), ("data_destination", approval["data_destination"])):
        _require(ticket.get(key) == value, f"admission ticket {key} differs from approved job")
    _require(0 < _positive(ticket.get("estimated_cost_usd"), "ticket estimate") <= approval["maximum_job_estimate_usd"], "ticket estimate exceeds per-job ceiling")
    return ticket


def _receipt(path: Path, plan_doc: dict, approval: dict, *, stage: str, job_ref: str, started: str, wall_s: float, exit_code: int, output_dir: Path, extra: dict):
    artifacts = []
    for item in sorted(output_dir.rglob("*")):
        if item.is_file() and item != path:
            artifacts.append({"path": str(item.relative_to(output_dir)), "sha256": _file_digest(item), "bytes": item.stat().st_size})
    _write(path, {"schema": "small-campaign-worker-receipt-v1", "campaign_id": plan_doc["settings"]["campaign_id"], "campaign_authorization_id": approval["campaign_authorization_id"], "plan_sha256": plan_doc["plan_sha256"], "approval_ref": approval["approval_ref"], "provider": approval["provider"], "stage": stage, "job_ref": job_ref, "started_at_utc": started, "ended_at_utc": _utc(), "wall_seconds": round(wall_s, 3), "exit_code": exit_code, "artifacts": artifacts, **extra})


def _run(argv: list[str], log: Path, cwd: Path | None = None) -> tuple[int, float, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = _utc()
    t0 = time.monotonic()
    with log.open("wb") as stream:
        try:
            rc = subprocess.run(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False).returncode
        except OSError as exc:
            stream.write(f"worker command could not start: {exc}\n".encode())
            rc = 127
    return rc, time.monotonic() - t0, started


def run_design(args):
    plan_doc, approval = _approved(args.plan, args.approval)
    settings = plan_doc["settings"]
    _require(settings["design"]["enabled"], "design is disabled")
    _require(args.seed in settings["design"]["seeds"], "seed outside approved plan")
    _require(plan_doc["jobs"] is None, "design launch requires a design-only plan")
    if args.seed != settings["design"]["seeds"][0]:
        _validate_smoke(args.smoke_receipt, plan_doc, "design")
    _require(IDENT.fullmatch(args.job_ref) is not None, "job reference is required")
    ticket = _admitted(args.ticket, plan_doc, approval, job_ref=args.job_ref, stage="design", arm=None, seed=args.seed)
    out = Path(args.out).resolve()
    _require(out == Path(settings["data_destination"]).resolve() or Path(settings["data_destination"]).resolve() in out.parents, "output is outside approved data destination")
    out.mkdir(parents=True, exist_ok=True)
    command = ["bindcraft", "design", settings["design"]["bindcraft_settings_path"], "--core", "benchmark", "--modality", "binder", "--set", f"campaign_seed={args.seed}"]
    rc, wall, started = _run(command, out / "bindcraft.log")
    selected = []
    ranked_rows = 0
    if rc == 0:
        try:
            project_folder = Path(_json(settings["design"]["bindcraft_settings_path"])["project_folder"])
            _require(project_folder.is_dir(), f"BindCraft project folder absent: {project_folder}")
            selected = sorted(p for p in project_folder.glob(settings["design"]["accepted_glob"]) if p.is_file())
            _require(bool(selected), "accepted structure glob matched no files")
            tables = list(project_folder.rglob("!_Ranked.csv"))
            _require(len(tables) == 1, "expected one 3_Ranked/!_Ranked.csv for the accepted designs")
            with tables[0].open(newline="", encoding="utf-8") as stream:
                ranked = list(csv.DictReader(stream))
            _require(bool(ranked) and {"design", "Binder_Sequence", "rank"} <= set(ranked[0]), "accepted ranked table is absent or lacks required columns")
            ranked_rows = len(ranked)
            _require(ranked_rows <= settings["design"]["maximum_candidates"], "ranked design count exceeds plan")
            for file in selected:
                destination = out / "accepted" / file.relative_to(project_folder)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, destination)
            for table_name in ("Ranked.csv", "!_Ranked.csv"):
                for table in project_folder.rglob(table_name):
                    destination = out / "accepted" / table.relative_to(project_folder)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(table, destination)
        except (ValueError, OSError, KeyError) as exc:
            with (out / "bindcraft.log").open("a", encoding="utf-8") as stream:
                stream.write(f"\naccepted-output harvest failed: {exc}\n")
            rc = 4
    _receipt(out / "worker-receipt.json", plan_doc, approval, stage="design", job_ref=args.job_ref, started=started, wall_s=wall, exit_code=rc, output_dir=out, extra={"seed": args.seed, "accepted_structures": len(selected), "ranked_designs": ranked_rows, "admission_ticket_sha256": ticket["record_hash"], "argv": command})
    return rc


def _validate_smoke(path: str | None, plan_doc: dict, stage: str, arm: str | None = None):
    _require(bool(path), "a completed smoke receipt is required before scale")
    receipt = _json(path)
    _require(receipt.get("plan_sha256") == plan_doc["plan_sha256"] and receipt.get("stage") == stage and receipt.get("exit_code") == 0, "smoke receipt does not match the plan and stage")
    if arm is not None:
        _require(receipt.get("arm") == arm and receipt.get("smoke") is True, "smoke receipt does not match the predictor arm")
        _require(receipt.get("parsed_structure_count", 0) >= receipt.get("candidate_count", 1) and receipt.get("parsed_pae_count", 0) >= receipt.get("candidate_count", 1), "smoke did not parse every expected structure and PAE artifact")
    else:
        _require(receipt.get("accepted_structures", 0) >= 1, "design smoke has no accepted structure")


def _write_boltz_inputs(jobs: list, msa_path: Path, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    _require(msa_path.is_file() and msa_path.stat().st_size > 0, "target MSA absent or empty")
    # The MSA path is absolute so Boltz resolves it from any worker cwd.
    for job in jobs:
        text = ("version: 1\nsequences:\n  - protein:\n      id: A\n"
                f"      sequence: {job['target']}\n      msa: {msa_path.resolve()}\n"
                "  - protein:\n      id: B\n"
                f"      sequence: {job['binder']}\n      msa: empty\n")
        (directory / f"{job['name']}.yaml").write_text(text, encoding="utf-8")


def _write_esmfold_inputs(jobs: list, seed: int, path: Path):
    payload = [{"id": job["name"], "sequences": [
        {"type": "protein", "id": "A", "sequence": job["target"], "msa": None},
        {"type": "protein", "id": "B", "sequence": job["binder"], "msa": None},
    ], "seeds": [seed], "num_diffusion_samples": 1} for job in jobs]
    _write(path, payload)


def _load_packaged_runtime():
    sibling_package = Path(__file__).resolve().parents[1] / "claude_binder"
    if sibling_package.is_dir():
        sys.path.insert(0, str(sibling_package.parent))


def _kit_verdict(log: Path, arm: str, mode: str, rc: int):
    _load_packaged_runtime()
    from claude_binder.kit_engagement import parse_engagement

    verdict = parse_engagement(
        requested_mode=mode,
        stdout=log.read_text(encoding="utf-8", errors="replace"),
        exit_code=rc,
    )
    _require(verdict.get("kit_id") == KITS[arm], f"kit announcement differs from requested arm {arm}")
    return verdict


def run_rescore(args):
    plan_doc, approval = _approved(args.plan, args.approval)
    settings = plan_doc["settings"]
    jobs = plan_doc.get("jobs")
    _require(jobs is not None, "rescore requires a plan bound to the candidate roster")
    _validate_jobs(jobs, settings)
    _require(args.arm in settings["rescore"]["predictors"], "arm outside approved plan")
    _require(args.seed in settings["rescore"]["seeds"], "seed outside approved plan")
    if args.smoke:
        _require(args.seed == settings["rescore"]["seeds"][0], "smoke must use the first declared seed")
    if not args.smoke:
        _validate_smoke(args.smoke_receipt, plan_doc, "rescore", args.arm)
    _require(IDENT.fullmatch(args.job_ref) is not None, "job reference is required")
    ticket = _admitted(args.ticket, plan_doc, approval, job_ref=args.job_ref, stage="rescore", arm=args.arm, seed=args.seed)
    out = Path(args.out).resolve()
    _require(out == Path(settings["data_destination"]).resolve() or Path(settings["data_destination"]).resolve() in out.parents, "output is outside approved data destination")
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        controls = {settings["controls"]["positive_id"], *settings["controls"]["negative_ids"]}
        candidates = [job for job in jobs if job["name"] not in controls]
        _require(bool(candidates), "smoke requires at least one candidate")
        jobs = [job for job in jobs if job["name"] in controls] + candidates[:1]
    target = settings["target"]
    if args.arm == "boltz2-kit":
        mode = settings["rescore"]["boltz2_mode"]
        _write_boltz_inputs(jobs, Path(target["msa_path"]), out / "inputs")
        command = ["bash", "run.sh", "pred", "--config", "h100", "--mode", mode,
                   "--input", str(out / "inputs"), "--out_dir", str(out / "predictions"),
                   "--recycling_steps", "3", "--diffusion_samples", "1", "--write_full_pae",
                   "--output_format", "mmcif"]
        command += (["--seed", str(args.seed)] if mode == "off" else ["--seeds", str(args.seed)])
        cwd = Path(args.kit_root or "/kit/boltz2")
    elif args.arm == "esmfold2-kit":
        mode = settings["rescore"]["esmfold2_mode"]
        _write_esmfold_inputs(jobs, args.seed, out / "inputs.json")
        command = ["bash", "run.sh", "pred", "--config", "h100", "--variant", settings["rescore"].get("esmfold2_variant", "fast"),
                   "--mode", mode, "--input", str(out / "inputs.json"), "--out_dir", str(out / "predictions"), "--seeds", str(args.seed)]
        if mode == "off":
            command += ["--backend", "fused"]
        cwd = Path(args.kit_root or "/kit/esmfold2")
    else:
        _require(args.arm == "esmfold2-platform", "unsupported arm")
        mode = "platform-stock"
        _write_esmfold_inputs(jobs, args.seed, out / "inputs.json")
        command = [sys.executable, str(Path(__file__).with_name("small_campaign_esmfold.py")),
                   "--input", str(out / "inputs.json"), "--out", str(out / "predictions"),
                   "--revision", settings["rescore"]["esmfold2_revision"]]
        cwd = None
    rc, wall, started = _run(command, out / "prediction.log", cwd=cwd)
    verdict = None
    if args.arm in KITS:
        try:
            verdict = _kit_verdict(out / "prediction.log", args.arm, mode, rc)
        except (ValueError, OSError, ImportError) as exc:
            verdict = {"accepted": False, "error": str(exc)}
        if not verdict.get("accepted", False):
            rc = 3
    structure_count = len(list((out / "predictions").rglob("*.cif")))
    pae_count = len(list((out / "predictions").rglob("*.npz")))
    if rc == 0 and (structure_count < len(jobs) or pae_count < len(jobs)):
        rc = 4
    parsed_structure_count = parsed_pae_count = 0
    if rc == 0 and args.smoke:
        _load_packaged_runtime()
        import numpy as np
        from claude_binder.adapters.binder_metrics import parse_structure_atoms
        try:
            for cif in (out / "predictions").rglob("*.cif"):
                _require(len(parse_structure_atoms(cif, argument="smoke_complex")) > 0, f"empty parsed structure: {cif}")
                parsed_structure_count += 1
            for npz in (out / "predictions").rglob("*.npz"):
                with np.load(npz, allow_pickle=False) as arrays:
                    if "pae" not in arrays:
                        continue
                    matrix = arrays["pae"]
                    _require(matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], f"PAE is not square: {npz}")
                    parsed_pae_count += 1
            _require(parsed_structure_count >= len(jobs) and parsed_pae_count >= len(jobs), "smoke could not parse every expected structure and PAE")
        except (ValueError, OSError) as exc:
            print(f"prediction smoke parser refused: {exc}", file=sys.stderr)
            rc = 4
    _receipt(out / "worker-receipt.json", plan_doc, approval, stage="rescore", job_ref=args.job_ref, started=started, wall_s=wall, exit_code=rc, output_dir=out, extra={"arm": args.arm, "mode": mode, "seed": args.seed, "smoke": args.smoke, "candidate_count": len(jobs), "structure_count": structure_count, "pae_artifact_count": pae_count, "parsed_structure_count": parsed_structure_count, "parsed_pae_count": parsed_pae_count, "kit_engagement": verdict, "admission_ticket_sha256": ticket["record_hash"], "argv": command})
    return rc


def score(args):
    """Score an explicit artifact manifest; mapping is never guessed from names."""
    rows = [row for path in args.predictions for row in _json(path)]
    _require(isinstance(rows, list) and rows, "prediction manifest must be a nonempty array")
    _load_packaged_runtime()
    import numpy as np
    from claude_binder.adapters.binder_metrics import (
        compute_ipsae, parse_structure_atoms, _kabsch, _apply_fit, _rmsd,
    )

    def ca_pose_rmsd(predicted, reference, target_chain, binder_chain, reference_target, reference_binder):
        pred = parse_structure_atoms(predicted, argument="predicted_cif")
        ref = parse_structure_atoms(reference, argument="design_pose")

        def paired_ca(pred_chain, ref_chain):
            lookup = {(r.seq_id, r.ins_code, r.comp_id): r for r in ref.chain_residues(ref_chain)}
            left, right = [], []
            for residue in pred.chain_residues(pred_chain):
                match = lookup.get((residue.seq_id, residue.ins_code, residue.comp_id))
                if match is None:
                    continue
                a = residue.named_atom_coords(("CA",))
                b = match.named_atom_coords(("CA",))
                if len(a) == len(b) == 1:
                    left.append(a[0]); right.append(b[0])
            _require(bool(left), f"no matched CA atoms for chain {pred_chain}")
            return np.asarray(left), np.asarray(right)

        target_pred, target_ref = paired_ca(target_chain, reference_target)
        binder_pred, binder_ref = paired_ca(binder_chain, reference_binder)
        fit = _kabsch(target_pred, target_ref)
        return _rmsd(_apply_fit(binder_pred, fit), binder_ref)

    scores = []
    for row in rows:
        cif = Path(row["complex_path"])
        pae_path = Path(row["pae_npz_path"])
        _require(cif.is_file() and pae_path.is_file(), f"prediction files absent for {row.get('candidate_id')}")
        with np.load(pae_path, allow_pickle=False) as arrays:
            _require("pae" in arrays, f"PAE key absent: {pae_path}")
            pae = arrays["pae"]
        orientation = row.get("pae_orientation")
        _require(orientation in ("aligned_rows", "aligned_columns"), "pae_orientation must be explicit")
        metric = compute_ipsae(pae, cif, row["target_chain"], row["binder_chain"], pae_orientation=orientation)
        pose = row.get("design_pose_path")
        if pose is not None:
            pose = Path(pose)
            _require(pose.is_file(), f"design pose absent: {pose}")
            pose_rmsd = ca_pose_rmsd(cif, pose, row["target_chain"], row["binder_chain"],
                                     row.get("reference_target_chain", row["target_chain"]),
                                     row.get("reference_binder_chain", row["binder_chain"]))
        else:
            pose_rmsd = None
        scores.append({"candidate_id": row["candidate_id"], "arm": row["arm"], "seed": row["seed"],
                       "ipsae_min": metric["ipsae_min"], "pose_rmsd": pose_rmsd,
                       "complex_path": str(cif), "complex_sha256": _file_digest(cif),
                       "pae_path": str(pae_path), "pae_sha256": _file_digest(pae_path),
                       "design_pose_path": str(pose) if pose else None,
                       "design_pose_sha256": _file_digest(pose) if pose else None,
                       "pae_orientation": orientation,
                       "job_ref": row["job_ref"]})
    _write(args.out, scores)


def _validated_provider_receipt(rec: dict, plan_doc: dict, approval: dict):
    job_id = rec.get("provider_job_id")
    job_ref = rec.get("job_ref")
    _require(isinstance(job_id, str) and job_id, "provider receipt requires an actual provider job ID")
    _require(isinstance(job_ref, str) and job_ref, "provider receipt requires a job reference")
    _require(rec.get("plan_sha256") == plan_doc["plan_sha256"], f"receipt {job_id} binds another plan")
    _require(rec.get("exit_code") == 0 and rec.get("status") == "completed", f"job {job_id} did not complete")
    worker_path = Path(str(rec.get("worker_receipt_path", "")))
    _require(worker_path.is_file(), f"worker receipt absent for {job_id}")
    _require(rec.get("worker_receipt_sha256") == _file_digest(worker_path), f"worker receipt digest mismatch for {job_id}")
    worker = _json(worker_path)
    _require(worker.get("plan_sha256") == plan_doc["plan_sha256"] and worker.get("job_ref") == job_ref and worker.get("exit_code") == 0, f"worker receipt contract failed for {job_id}")
    ticket_path = Path(str(rec.get("admission_ticket_path", "")))
    _require(ticket_path.is_file(), f"admission ticket absent for {job_id}")
    ticket = _admitted(str(ticket_path), plan_doc, approval, job_ref=job_ref,
                       stage=worker.get("stage"), arm=worker.get("arm"), seed=worker.get("seed"))
    _require(worker.get("admission_ticket_sha256") == ticket["record_hash"], f"worker admission ticket mismatch for {job_id}")
    wall = _positive(rec.get("wall_seconds"), f"receipt {job_id}.wall_seconds")
    settled = rec.get("settled_cost_usd")
    if settled is not None:
        settled = _positive(settled, f"receipt {job_id}.settled_cost_usd")
    rate = rec.get("list_rate_usd_per_hour")
    billable = rec.get("billable_seconds", wall)
    estimate = None if rate is None else _positive(rate, f"receipt {job_id}.list_rate_usd_per_hour") * _positive(billable, f"receipt {job_id}.billable_seconds") / 3600
    start = rec.get("started_at_utc", worker.get("started_at_utc"))
    end = rec.get("ended_at_utc", worker.get("ended_at_utc"))
    _require(isinstance(start, str) and isinstance(end, str), f"receipt {job_id} lacks start/end timestamps")
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    _require(start_dt.tzinfo is not None and end_dt.tzinfo is not None and end_dt >= start_dt, f"receipt {job_id} timestamps invalid")
    return {"job_id": job_id, "job_ref": job_ref, "worker": worker, "worker_receipt_path": str(worker_path), "wall_seconds": wall,
            "settled_cost_usd": settled, "list_rate_estimate_usd": estimate,
            "start": start_dt, "end": end_dt}


def report(args):
    plan_doc, approval = _approved(args.plan, args.approval)
    settings = plan_doc["settings"]
    jobs = plan_doc.get("jobs")
    _require(jobs is not None, "report requires a roster-bound plan")
    _validate_jobs(jobs, settings)
    rows = _json(args.scores)
    receipts = _json(args.receipts)
    _require(isinstance(rows, list) and isinstance(receipts, list), "scores and receipts must be arrays")
    arms = settings["rescore"]["predictors"]
    seeds = settings["rescore"]["seeds"]
    ranking_rule = settings["rescore"].get("ranking_rule", DEFAULT_RANKING_RULE)
    by_key = {}
    for row in rows:
        key = (row.get("candidate_id"), row.get("arm"), row.get("seed"))
        _require(key not in by_key, f"duplicate score: {key}")
        _require(key[0] in {j["name"] for j in jobs} and key[1] in arms and key[2] in seeds, f"score outside plan: {key}")
        _require(0 <= _positive(row.get("ipsae_min"), f"{key}.ipsae_min") <= 1, f"ipSAE outside 0..1: {key}")
        _require(Path(str(row.get("complex_path", ""))).is_file(), f"structure absent: {key}")
        _require(row.get("complex_sha256") == _file_digest(Path(row["complex_path"])), f"structure digest mismatch: {key}")
        _require(row.get("pae_sha256") == _file_digest(Path(row["pae_path"])), f"PAE digest mismatch: {key}")
        _require(row.get("pae_orientation") in ("aligned_rows", "aligned_columns"), f"PAE orientation missing for {key}")
        if row.get("design_pose_path") is not None:
            _require(row.get("design_pose_sha256") == _file_digest(Path(row["design_pose_path"])), f"design pose digest mismatch: {key}")
        by_key[key] = row
    expected = {(j["name"], arm, seed) for j in jobs for arm in arms for seed in seeds}
    _require(set(by_key) == expected, f"{len(seeds)}-seed two-arm score matrix incomplete: {len(expected - set(by_key))} missing")
    provider_receipts = {}
    total_cost = 0.0
    total_wall = 0.0
    list_rate_estimate = 0.0
    missing_rate = []
    unsettled = []
    starts = []
    ends = []
    for rec in receipts:
        item = _validated_provider_receipt(rec, plan_doc, approval)
        _require(item["job_ref"] not in provider_receipts, "provider receipts need unique job references")
        total_wall += item["wall_seconds"]
        starts.append(item["start"]); ends.append(item["end"])
        if item["settled_cost_usd"] is None:
            unsettled.append(item["job_id"])
        else:
            total_cost += item["settled_cost_usd"]
        if item["list_rate_estimate_usd"] is None:
            missing_rate.append(item["job_id"])
        else:
            list_rate_estimate += item["list_rate_estimate_usd"]
        provider_receipts[item["job_ref"]] = item
    _require(all(row["job_ref"] in provider_receipts for row in rows), "a score lacks a completed provider receipt")
    for row in rows:
        worker = provider_receipts[row["job_ref"]]["worker"]
        _require(worker.get("stage") == "rescore" and worker.get("arm") == row["arm"] and worker.get("seed") == row["seed"], "score arm/seed differs from worker receipt")
        artifacts = worker.get("artifacts")
        _require(isinstance(artifacts, list) and artifacts, "rescore worker receipt lacks artifact inventory")
        worker_root = Path(provider_receipts[row["job_ref"]]["worker_receipt_path"]).parent
        artifact_by_path = {}
        for item in artifacts:
            _require(isinstance(item, dict) and isinstance(item.get("path"), str) and isinstance(item.get("sha256"), str), "worker artifact entry lacks path or digest")
            artifact_by_path[(worker_root / item["path"]).resolve()] = item["sha256"]
        _require(artifact_by_path.get(Path(row["complex_path"]).resolve()) == row["complex_sha256"], "scored complex is absent from worker artifacts or has another digest")
        _require(artifact_by_path.get(Path(row["pae_path"]).resolve()) == row["pae_sha256"], "scored PAE is absent from worker artifacts or has another digest")
    if settings["design"]["enabled"]:
        _require(bool(args.linked_phase), "design-enabled campaign report requires --linked-phase with design plan, approval, and receipts")
        linked = _json(args.linked_phase)
        design_plan, design_approval = _approved(linked["plan"], linked["approval"])
        _require(design_plan["jobs"] is None and design_plan["settings"]["campaign_id"] == settings["campaign_id"], "linked phase must be this campaign's design plan")
        _require(design_approval["campaign_authorization_id"] == approval["campaign_authorization_id"], "linked design phase has another campaign authorization")
        _require(approval["maximum_spend_usd"] <= design_approval["maximum_spend_usd"], "rescore approval increases design campaign ceiling")
        design_receipts = _json(linked["receipts"])
        _require(isinstance(design_receipts, list) and design_receipts, "linked design receipts are required")
        for rec in design_receipts:
            item = _validated_provider_receipt(rec, design_plan, design_approval)
            _require(item["worker"].get("stage") == "design" and item["worker"].get("accepted_structures", 0) >= 1, "linked design receipt has no accepted structure")
            _require(item["job_ref"] not in provider_receipts, "job reference reused across phases")
            total_wall += item["wall_seconds"]
            starts.append(item["start"]); ends.append(item["end"])
            if item["settled_cost_usd"] is None:
                unsettled.append(item["job_id"])
            else:
                total_cost += item["settled_cost_usd"]
            if item["list_rate_estimate_usd"] is None:
                missing_rate.append(item["job_id"])
            else:
                list_rate_estimate += item["list_rate_estimate_usd"]
            provider_receipts[item["job_ref"]] = item
    _require(total_cost <= approval["maximum_spend_usd"], "settled component exceeds approved spend ceiling")
    pos = settings["controls"]["positive_id"]
    negatives = settings["controls"]["negative_ids"]
    control_summary = {}
    calibrated = {}
    for arm in arms:
        gate = settings["rescore"]["gates"][arm]
        p = sum(by_key[(pos, arm, seed)]["ipsae_min"] for seed in seeds) / len(seeds)
        negative_means = {cid: sum(by_key[(cid, arm, seed)]["ipsae_min"] for seed in seeds) / len(seeds) for cid in negatives}
        negative_bests = {cid: max(by_key[(cid, arm, seed)]["ipsae_min"] for seed in seeds) for cid in negatives}
        n = max(negative_bests.values()) if gate["negative_reference"] == "largest_negative_best_seed" else max(negative_means.values())
        _require(p - n >= gate["minimum_positive_margin"] and p > n, f"{arm}: positive control does not separate from the named negative reference")
        control_summary[arm] = {"positive_mean_ipsae": p, "negative_means": negative_means, "negative_best_seeds": negative_bests, "negative_reference": n, "margin": p-n, "gate": gate}
        calibrated[arm] = (n, p)
    control_ids = {pos, *negatives}
    ranked = []
    for job in jobs:
        cid = job["name"]
        if cid in control_ids:
            continue
        per_arm = {}
        normalized = []
        bests = []
        for arm in arms:
            values = [by_key[(cid, arm, seed)]["ipsae_min"] for seed in seeds]
            poses = [by_key[(cid, arm, seed)].get("pose_rmsd") for seed in seeds]
            _require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in poses), f"{cid}/{arm}: pose RMSD missing")
            mean = sum(values) / len(values)
            gate = settings["rescore"]["gates"][arm]
            passed = mean > gate["candidate_multiplier"] * calibrated[arm][0] and max(poses) <= gate["maximum_pose_rmsd"]
            low, high = calibrated[arm]
            normalized.append((mean-low)/(high-low))
            bests.append(max(values))
            per_arm[arm] = {"mean_ipsae": mean, "best_ipsae": bests[-1], "seed_min": min(values), "seed_max": bests[-1], "max_pose_rmsd": max(poses), "candidate_threshold": gate["candidate_multiplier"] * calibrated[arm][0], "passed": passed, "control_normalized": normalized[-1]}
        components = bests if ranking_rule == "mean_of_arm_best_ipsae" else normalized
        ranked.append({"candidate_id": cid, "passed": all(v["passed"] for v in per_arm.values()), "rank_score": sum(components)/len(components), "arms": per_arm})
    ranked.sort(key=lambda row: (not row["passed"], -row["rank_score"], row["candidate_id"]))
    output = Path(args.out)
    csv_path = output.with_suffix(".csv")
    fasta_path = output.with_suffix(".fasta")
    summary_path = output.with_suffix(".txt")
    elapsed = (max(ends) - min(starts)).total_seconds() if starts else None
    seed_description = ", ".join(str(seed) for seed in seeds)
    result = {"schema": "small-campaign-report-v1", "campaign_id": settings["campaign_id"], "campaign_authorization_id": approval["campaign_authorization_id"], "plan_sha256": plan_doc["plan_sha256"], "approval_ref": approval["approval_ref"], "rescore_seeds": seeds, "ranking_rule": ranking_rule, "controls": control_summary, "ranking": ranked, "cost": {"scope": "design + rescore" if args.linked_phase else "rescore only", "provider_job_count": len(provider_receipts), "settled_usd": total_cost if not unsettled else None, "settled_component_usd": total_cost, "unsettled_job_ids": unsettled, "list_rate_estimate_usd": list_rate_estimate if not missing_rate else None, "list_rate_component_usd": list_rate_estimate, "missing_list_rate_job_ids": missing_rate, "elapsed_campaign_seconds": elapsed, "provider_wall_seconds_sum": total_wall}, "exports": {"ranked_csv": str(csv_path), "shortlist_fasta": str(fasta_path), "summary_text": str(summary_path)}, "claim": f"computational shortlist; two predictor routes, {len(seeds)} seed-labeled prediction calls per route (seed IDs: {seed_description}); no binding or specificity validation"}
    _write(output, result)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rank", "candidate_id", "passed", "ranking_rule", "rank_score", *[f"{arm}_mean_ipsae" for arm in arms], *[f"{arm}_best_ipsae" for arm in arms], *[f"{arm}_max_pose_rmsd" for arm in arms]])
        for index, item in enumerate(ranked, 1):
            writer.writerow([index, item["candidate_id"], item["passed"], ranking_rule, f"{item['rank_score']:.6f}", *[f"{item['arms'][arm]['mean_ipsae']:.6f}" for arm in arms], *[f"{item['arms'][arm]['best_ipsae']:.6f}" for arm in arms], *[f"{item['arms'][arm]['max_pose_rmsd']:.3f}" for arm in arms]])
    sequence_by_id = {job["name"]: job["binder"] for job in jobs}
    with fasta_path.open("w", encoding="utf-8") as stream:
        for item in ranked:
            if item["passed"]:
                stream.write(f">{item['candidate_id']} rank_score={item['rank_score']:.6f}\n{sequence_by_id[item['candidate_id']]}\n")
    ranking_description = ("raw mean of the two per-arm best-of-seeds ipSAE_min values"
                           if ranking_rule == "mean_of_arm_best_ipsae"
                           else "mean of two per-arm control-normalized mean ipSAE_min values")
    lines = [f"Campaign: {settings['campaign_id']}", f"Plan: {plan_doc['plan_sha256']}", f"Approval: {approval['approval_ref']}", f"Rescore: {len(seeds)} seed IDs per predictor ({seed_description})", f"Ranking rule: {ranking_rule} ({ranking_description})", "Positive and candidate ipSAE gates use seed means; candidate pose gates use maximum RMSD across seeds.", "Computational shortlist only; binding and specificity are unvalidated.", "", "Control calibration:"]
    for arm in arms:
        control = control_summary[arm]
        lines.append(f"  {arm}: positive mean ipSAE_min={control['positive_mean_ipsae']:.4f}; negative reference={control['negative_reference']:.4f}; margin={control['margin']:.4f}; rule={control['gate']['rule_context']}")
    lines += ["", f"Passed candidates: {sum(item['passed'] for item in ranked)}/{len(ranked)}", f"Elapsed campaign time: {elapsed:.1f} s" if elapsed is not None else "Elapsed campaign time: unavailable", f"Provider wall time summed across receipts: {total_wall:.1f} s", f"Settled component: ${total_cost:.4f}"]
    lines.append("Campaign settled cost: pending (unsettled jobs: " + ", ".join(unsettled) + ")" if unsettled else f"Campaign settled cost: ${total_cost:.4f}")
    lines.append("List-rate estimate: unavailable (missing rates for " + ", ".join(missing_rate) + ")" if missing_rate else f"List-rate estimate from receipt billable time and rate: ${list_rate_estimate:.4f}")
    if not args.linked_phase:
        lines.append("No linked design phase was included.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bundle-worker", help="pack the worker script and bundled runtime for provider input staging")
    p.add_argument("--out", required=True); p.set_defaults(func=bundle_worker)
    p = sub.add_parser("verify-worker", help="verify an extracted worker bundle before running a paid job")
    p.add_argument("--root", required=True); p.add_argument("--archive"); p.add_argument("--expected-sha256"); p.set_defaults(func=verify_worker)
    p = sub.add_parser("plan", help="validate settings and optionally bind a candidate roster")
    p.add_argument("--settings", required=True); p.add_argument("--jobs"); p.add_argument("--out", required=True); p.set_defaults(func=plan)
    p = sub.add_parser("make-roster", help="extract accepted BindCraft designs and append explicit controls")
    for name in ("settings", "ranked-csv", "controls", "out", "design-target-chain", "design-binder-chain"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--limit", type=int, required=True); p.set_defaults(func=make_roster)
    p.add_argument("--target-state-suffix", help="explicit suffix such as _targetA when ranked design has multiple accepted complexes")
    p = sub.add_parser("index-artifacts", help="map one arm/seed's named structures and PAE files")
    for name in ("plan", "arm", "job-ref", "prediction-root", "target-chain", "binder-chain", "reference-target-chain", "reference-binder-chain", "pae-orientation", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, required=True)
    p._option_string_actions["--pae-orientation"].choices = ("aligned_rows", "aligned_columns")
    p.set_defaults(func=index_artifacts)
    p = sub.add_parser("admit", help="reserve one estimated job cost under a locked campaign ledger before provider submission")
    for name in ("plan", "approval", "ledger", "job-ref", "stage", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--arm", choices=ARMS); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--estimate-usd", type=float, required=True); p.set_defaults(func=admit)
    p = sub.add_parser("run-design", help="run one approved BindCraft2 seed inside a worker")
    for name in ("plan", "approval", "ticket", "out", "job-ref"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, required=True); p.set_defaults(func=run_design)
    p.add_argument("--smoke-receipt")
    p = sub.add_parser("run-rescore", help="run one approved predictor arm and seed inside a worker")
    for name in ("plan", "approval", "ticket", "out", "job-ref"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--arm", choices=ARMS, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--kit-root"); p.set_defaults(func=run_rescore)
    p.add_argument("--smoke", action="store_true"); p.add_argument("--smoke-receipt")
    p = sub.add_parser("score", help="compute ipSAE from an explicit CIF/PAE artifact manifest")
    p.add_argument("--predictions", nargs="+", required=True); p.add_argument("--out", required=True); p.set_defaults(func=score)
    p = sub.add_parser("report", help="gate controls, poses, score matrix, and provider receipts; rank candidates")
    for name in ("plan", "approval", "scores", "receipts", "out"):
        p.add_argument("--" + name, required=True)
    p.set_defaults(func=report)
    p.add_argument("--linked-phase", help="JSON file naming the design plan, approval and provider receipts")
    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"small campaign refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
