"""Run the deterministic first-run fixture from an installed skill directory."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from claude_binder.paths import package_root


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPOSITORY_ROOT / "scripts" / "build_claude_binder_skill.py"
KERNEL_FREE_RUN = r'''
import argparse
import json
import os
import sys
from pathlib import Path

skill_root = Path(sys.argv[1]).resolve()
work_root = Path(sys.argv[2]).resolve()
assert "PYTHONPATH" not in os.environ
assert "/opt/homebrew/bin" not in os.environ.get("PATH", "").split(os.pathsep)

kernel_path = skill_root / "kernel.py"
session_globals = {"__name__": "__main__"}
exec(compile(kernel_path.read_text(encoding="utf-8"), str(kernel_path), "exec"), session_globals)

version = session_globals["binder_version"]()
assert version["source"] == "sibling-files", version
assert Path(version["path"]) == skill_root / "claude_binder" / "__init__.py", version

# The kernel can bind the sibling package without sys.path. Child stage commands
# need this explicit package parent after PYTHONSAFEPATH removes implicit entries.
os.environ["PYTHONPATH"] = str(skill_root)
package_root = skill_root / "claude_binder"

from claude_binder import lane
from claude_binder.adapters import browser_renderer

profile = lane.load_profile(
    package_root / "data" / "templates" / "profiles" / "local-contract-test.json"
)
viewer_stage = next(stage for stage in profile["stages"] if stage["stage_id"] == "render-viewer")
assert viewer_stage["adapter_id"] == "browser-viewer"

structure_path = (
    package_root
    / "data"
    / "roster-evidence"
    / "canary"
    / "rf3"
    / "structures"
    / "design-spec_pdl1_binder_0_model_0.cif"
)
assert structure_path.is_file()
run_root = work_root / "free-viewer-run"
artifacts = run_root / "artifacts"
(artifacts / "scores").mkdir(parents=True)
(run_root / "status.json").write_text(json.dumps({"run_id": "installed-free-viewer"}), encoding="utf-8")
(run_root / "runtime-config.resolved.json").write_text(
    json.dumps(
        {
            "targets": [
                {
                    "target_id": "fixture-target",
                    "structure_path": str(structure_path),
                    "site": {
                        "design_residues": ["A:1"],
                        "reference_contact_residues": [],
                        "contact_cutoff_angstrom": 5.0,
                    },
                }
            ]
        }
    ),
    encoding="utf-8",
)
(artifacts / "scores" / "ranked-candidates.json").write_text(
    json.dumps(
        {
            "campaign_id": "installed-free-viewer",
            "primary_target_id": "fixture-target",
            "ranked_candidates": [
                {
                    "rank": 1,
                    "candidate_id": "fixture-candidate",
                    "target_id": "fixture-target",
                    "generator": "fixture-generator",
                    "sequence_length": 3,
                    "rank_score": 0.9,
                    "ipsae_min_ensemble": 0.8,
                    "sc_dockq_ensemble": 0.7,
                    "selected_seed_by_predictor": {"fixture-predictor": 0},
                }
            ],
        }
    ),
    encoding="utf-8",
)
(artifacts / "scores" / "uniform-observations.jsonl").write_text(
    json.dumps(
        {
            "candidate_id": "fixture-candidate",
            "predictor": "fixture-predictor",
            "seed": 0,
            "predicted_complex_path": str(structure_path),
            "binder_chain_id": "B",
            "target_chain_id": "A",
            "ipsae_min": 0.8,
            "sc_dockq": 0.7,
        }
    )
    + "\n",
    encoding="utf-8",
)

viewer_root = work_root / "viewer"
assert browser_renderer.run(argparse.Namespace(run_dir=run_root, out_dir=viewer_root)) == 0
viewer_page = viewer_root / "index.html"
thumbnails = sorted((viewer_root / "thumbnails").glob("*.png"))
assert viewer_page.is_file()
assert viewer_page.stat().st_size > 1_024
assert thumbnails
assert all(thumbnail.stat().st_size > 1_024 for thumbnail in thumbnails)
assert any("fixture-candidate" in thumbnail.name for thumbnail in thumbnails)
print(json.dumps({
    "viewer_page": str(viewer_page),
    "viewer_page_bytes": viewer_page.stat().st_size,
    "thumbnail_count": len(thumbnails),
    "thumbnail_bytes": thumbnails[0].stat().st_size,
}))
'''


def _hostile_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONSAFEPATH"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PATH"] = os.pathsep.join(
        item for item in environment.get("PATH", "").split(os.pathsep)
        if item and item != "/opt/homebrew/bin"
    )
    return environment


def test_shipped_campaigns_declare_non_commercial_use() -> None:
    campaign_paths = []
    for path in sorted((package_root() / "data").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "campaign_id" in payload:
            campaign_paths.append(path)
            assert payload.get("declared_use") == "non-commercial", path
    assert campaign_paths


@pytest.mark.skipif(
    not BUILD_SCRIPT.is_file(),
    reason=(
        "the build script is a contributor tool and does not ship, so an installed "
        "skill running its own evidence tests cannot rebuild itself"
    ),
)
def test_installed_skill_emits_the_free_viewer_page(tmp_path: Path) -> None:
    skill_root = tmp_path / "installed" / "claude-binder-lane"
    build = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--output", str(skill_root)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    foreign_cwd = tmp_path / "foreign-working-directory"
    foreign_cwd.mkdir()
    result = subprocess.run(
        [sys.executable, "-S", "-c", KERNEL_FREE_RUN, str(skill_root), str(tmp_path / "run")],
        cwd=foreign_cwd,
        env=_hostile_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads(result.stdout.splitlines()[-1])
    assert Path(summary["viewer_page"]).is_file()
    assert summary["viewer_page_bytes"] > 1_024
    assert summary["thumbnail_count"] > 0
    assert summary["thumbnail_bytes"] > 1_024
