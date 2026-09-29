"""Focused public evidence for partial FINAL seed coverage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from claude_binder import lane
from claude_binder.paths import package_root

TEMPLATE_ROOT = package_root() / "data" / "templates"
CAMPAIGN_TEMPLATE = TEMPLATE_ROOT / "campaign.template.json"
FULL_PROFILE = TEMPLATE_ROOT / "profiles" / "full-ensemble.template.json"
LOCAL_CAMPAIGN = package_root() / "data" / "fixtures" / "local-contract" / "campaign.json"

def replace_required(value):
    if isinstance(value, dict):
        return {key: replace_required(child) for key, child in value.items()}
    if isinstance(value, list):
        return [replace_required(child) for child in value]
    if isinstance(value, str) and "URL for your" in value:
        return "https://fal.run/example/fixture"
    if isinstance(value, str) and "__REQUIRED__" in value:
        return value.replace("__REQUIRED__", "fixture")
    return value

def fixture_config(structure_path: Path, residue_map_path: Path) -> dict:
    campaign = json.loads(CAMPAIGN_TEMPLATE.read_text())
    profile = json.loads(FULL_PROFILE.read_text())
    config = replace_required({**campaign, **{key: value for key, value in profile.items() if key not in {"schema_version", "template_id"}}})
    # The template requires an explicit use declaration, and the generic
    # placeholder substitution above cannot guess one. A fixture campaign is
    # non-commercial, so it says so, exactly as a real campaign must.
    config["declared_use"] = "non-commercial"
    config["campaign_id"] = "fixture-binder-campaign"
    config["run_id"] = "fixture-binder-run"
    config["targets"][0]["source_id"] = "fixture-target"
    config["targets"][0]["structure_path"] = str(structure_path)
    config["targets"][0]["chains"][0]["chain_id"] = "A"
    config["targets"][0]["entities"][0]["chain_ids"] = ["A"]
    config["targets"][0]["site"]["design_residues"] = ["A:1-3"]
    config["targets"][0]["site"]["reference_contact_residues"] = ["A:1", "A:2"]
    config["targets"][0]["site"]["residue_map_path"] = str(residue_map_path)
    config["binder"]["target_chain_id"] = "A"
    config["binder"]["binder_chain_id"] = "B"
    config["binder"]["minimum_length"] = 50
    config["binder"]["maximum_length"] = 120
    # This shared fixture enables two optimization rounds, so it must carry the
    # same explicit stopping decision required of a real multi-round campaign.
    config["optimization"]["early_stop_margin"] = 0.029008
    config["optimization"]["early_stop_margin_metric"] = "ipsae_min"
    for group in ("positive", "negative"):
        config["controls"][group][0]["structure_path"] = str(structure_path)
        config["controls"][group][0]["target_chain"] = "A"
        config["controls"][group][0]["binder_chain"] = "B"
    config["controls"]["positive"][0]["gates"] = [
        {"metric": "ipsae_min", "operator": "minimum", "threshold": 0.8},
        {"metric": "sc_dockq", "operator": "minimum", "threshold": 0.1},
        {"metric": "site_contact_iou", "operator": "minimum", "threshold": 0.1},
        {"metric": "target_contact_recall", "operator": "minimum", "threshold": 0.1},
    ]
    config["controls"]["negative"][0]["gates"] = [
        {"metric": "ipsae_min", "operator": "maximum", "threshold": 0.2},
        {"metric": "site_contact_iou", "operator": "maximum", "threshold": 0.1},
    ]
    config["selection"].update(
        {
            "final_count": 3,
            "minimum_generators": 2,
            "maximum_fraction_per_generator": 0.5,
        }
    )
    config["scoring"]["thresholds"] = {
        "minimum_ipsae_min_ensemble": 0.1,
        "minimum_sc_dockq_ensemble": 0.1,
        "minimum_site_contact_iou": 0.1,
        "minimum_target_contact_recall": 0.1,
        "maximum_clash_count": 10,
        "positive_control_minimum_ipsae_min": 0.8,
        "positive_control_minimum_sc_dockq": 0.1,
        "positive_control_minimum_site_contact_iou": 0.1,
        "positive_control_minimum_target_contact_recall": 0.1,
        "negative_control_maximum_ipsae_min": 0.2,
    }
    config["scoring"]["implementations"] = {
        "ipsae_revision": "fixture-ipsae-v1",
        "ipsae_interface_cutoff_angstrom": 10.0,
        "dockq_revision": "fixture-dockq-v1",
        "site_scorer_revision": "fixture-site-scorer-v1",
        "site_metric_basis": "selected-cofold-pose",
        "minimum_aligned_target_residues": 20,
        "maximum_target_alignment_rmsd": 2.0,
    }
    config["filters"] = json.loads(LOCAL_CAMPAIGN.read_text())["filters"]
    return config

def observation(
    candidate_id: str,
    generator: str,
    predictor: str,
    seed: int,
    ipsae: float,
    dockq: float,
    control_type: str = "candidate",
) -> dict:
    return {
        "target_id": "primary-target",
        "candidate_id": candidate_id,
        "generator": generator,
        "predictor": predictor,
        "score_instrument": lane.score_instrument_arm_name(predictor),
        "seed": seed,
        "phase": "uniform-rescore",
        "attempt_id": "fixture-attempt",
        "model_revision": "fixture",
        "control_type": control_type,
        "control_role": (
            "known-same-site-complex"
            if control_type == "positive"
            else "matched-wrong-pair" if control_type == "negative" else None
        ),
        "control_structure_sha256": "a" * 64 if control_type != "candidate" else None,
        "status": "scored",
        "filter_pass": True,
        "target_sha256": "a" * 64,
        "sequence_sha256": (candidate_id[0].encode().hex() * 64)[:64],
        "target_chain_id": "A",
        "binder_chain_id": "B",
        "predicted_target_chain_id": "A",
        "predicted_binder_chain_id": "B",
        "reference_target_chain_id": "A",
        "reference_binder_chain_id": "B",
        "design_pose_path": f"structures/{candidate_id}-design.pdb",
        "design_pose_sha256": (candidate_id[-1].encode().hex() * 64)[:64],
        "predicted_complex_path": f"complexes/{candidate_id}-{predictor}-{seed}.pdb",
        "predicted_complex_sha256": ((predictor[0] + str(seed)).encode().hex() * 64)[:64],
        "pae_path": f"confidence/{candidate_id}-{predictor}-{seed}.json",
        "pae_sha256": "c" * 64,
        "metric_source_path": f"metrics/{candidate_id}-{predictor}-{seed}.json",
        "metric_source_sha256": "d" * 64,
        "raw_prediction_record_sha256": "e" * 64,
        "chain_mapping": {"target": "A", "binder": "B"},
        "aligned_target_residue_count": 100,
        "target_alignment_rmsd": 0.4,
        "ipsae_implementation_revision": "fixture-ipsae-v1",
        "ipsae_interface_cutoff_angstrom": 10.0,
        "dockq_implementation_revision": "fixture-dockq-v1",
        "site_scorer_revision": "fixture-site-scorer-v1",
        "site_residue_map_sha256": "b" * 64,
        "site_contact_cutoff_angstrom": 5.0,
        "site_atom_selection": "heavy-atoms",
        "site_metric_basis": "selected-cofold-pose",
        "ipsae_target_to_binder": ipsae,
        "ipsae_binder_to_target": ipsae,
        "ipsae_min": ipsae,
        "sc_dockq": dockq,
        "dockq": dockq,
        "fnat": dockq,
        "interface_rmsd": 1.0,
        "ligand_rmsd": 1.5,
        "mapping_status": "ok",
        "site_contact_iou": 0.0 if control_type == "negative" else 0.7,
        "target_contact_recall": 0.0 if control_type == "negative" else 0.8,
        "target_contact_precision": 0.75,
        "hotspot_recovery": 0.8,
        "offsite_contact_fraction": 0.1,
        "iptm": ipsae,
        "interface_pae": 4.0,
        "interface_plddt": 82.0,
        "clash_count": 0,
        "contact_count": 20,
    }

class PartialFinalSeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        structure = root / "target.pdb"
        structure.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C\n"
            "ATOM      2  CA  GLY B   1       5.000   0.000   0.000  1.00 80.00           C\nEND\n"
        )
        residue_map = root / "residue-map.json"
        residue_map.write_text('{}')
        config = fixture_config(structure, residue_map)
        config["cofold"]["predictors"] = [
            item for item in config["cofold"]["predictors"] if item["id"] == "esmfold2"
        ]
        config["cofold"]["rescore_seeds"] = list(range(5))
        config["scoring"]["minimum_seed_observations"] = 5
        config["selection"].update(
            final_count=1, minimum_generators=1, maximum_fraction_per_generator=1.0
        )
        self.config = config

    def rows(self, partial_seed_count: int = 3) -> list[dict]:
        rows = [
            observation("complete", "rfdiffusion", "esmfold2", seed, 0.5, 0.5)
            for seed in range(5)
        ]
        rows.extend(
            observation("partial", "rfdiffusion", "esmfold2", seed, 0.9, 0.9)
            for seed in range(partial_seed_count)
        )
        for seed in range(partial_seed_count, 5):
            failed = observation("partial", "rfdiffusion", "esmfold2", seed, 0.9, 0.9)
            failed.update(status="failed", failure_code="fold-failed", failure_reason="predictor failed")
            rows.append(failed)
        rows.extend(
            observation("positive-reference", "control", "esmfold2", seed, 0.9, 0.9, "positive")
            for seed in range(5)
        )
        rows.extend(
            observation("matched-negative", "control", "esmfold2", seed, 0.1, 0.1, "negative")
            for seed in range(5)
        )
        return rows

    def test_partial_seed_row_stays_in_final_sheet_below_complete_row(self) -> None:
        result = lane.rank_candidates(self.config, self.rows())
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual([row["candidate_id"] for row in result["ranked_candidates"]], ["complete", "partial"])
        complete, partial = result["ranked_candidates"]
        self.assertEqual(complete["rank_status"], "ranked")
        self.assertEqual(partial["rank_status"], "partial")
        self.assertEqual(partial["coverage"]["esmfold2"]["scored_seed_count"], 3)
        self.assertEqual(partial["coverage"]["esmfold2"]["observed_seed_count"], 5)
        self.assertEqual(partial["coverage"]["esmfold2"]["required_seeds"], list(range(5)))
        self.assertFalse(partial["coverage_complete"])
        self.assertFalse(partial["eligible"])
        self.assertGreater(partial["rank_score"], complete["rank_score"])
        self.assertEqual(result["selected_candidates"][0]["candidate_id"], "complete")
        self.assertEqual(result["ranking_receipt"]["partial_candidates"][0]["scored_seed_count_by_predictor"], {"esmfold2": 3})
        self.assertEqual(result["unranked_candidate_count"], 0)

    def test_partial_only_sheet_cannot_claim_a_winner(self) -> None:
        rows = [row for row in self.rows() if row["candidate_id"] != "complete"]
        result = lane.rank_candidates(self.config, rows)
        self.assertTrue(result["ok"], result["errors"])
        self.assertEqual(result["ranked_candidates"][0]["rank_status"], "partial")
        self.assertEqual(result["eligible_candidate_count"], 0)
        self.assertEqual(result["selected_candidates"], [])
        self.assertEqual(result["best_design_claim_status"], "unavailable")

    def test_missing_negative_control_still_refuses_final_rank(self) -> None:
        rows = [row for row in self.rows() if row["control_type"] != "negative"]
        result = lane.rank_candidates(self.config, rows)
        self.assertFalse(result["ok"])
        self.assertTrue(any("control IDs" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
