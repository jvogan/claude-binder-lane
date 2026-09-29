from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from claude_binder import lane
from claude_binder.adapters import optimization_controller as controller
from claude_binder.adapters import promotion_selector
from claude_binder.paths import package_root


# claude_binder/data/model-roster.json records this distinct-seed spread.
MEASURED_EARLY_STOP_MARGIN = 0.029008


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def score_row(candidate_id: str, score: float) -> dict:
    return {
        "target_id": "target",
        "candidate_id": candidate_id,
        "predictor": "arm",
        "seed": 0,
        "control_type": "candidate",
        "generator": "test",
        "design_pose_path": "test.pdb",
        "filter_pass": True,
        "ipsae_min": score,
        "sc_dockq": score,
        "site_contact_iou": score,
        "target_contact_recall": score,
        "target_contact_precision": score,
        "hotspot_recovery": score,
        "offsite_contact_fraction": 0.0,
        "clash_count": 0,
        "contact_count": 1,
        "iptm": score,
        "interface_pae": 1.0,
        "interface_plddt": 80.0,
        "status": "scored",
    }


def rank_rows(_config: dict, observations: list[dict], _seeds: list[int]) -> list[dict]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "eligible": True,
            "coverage_complete": True,
            "filter_pass": True,
            "ipsae_min_ensemble": row["ipsae_min"],
        }
        for row in observations
        if row.get("control_type") == "candidate"
    ]


def controller_config(*, rounds: int) -> dict:
    return {
        "cofold": {"screen_seeds": [0]},
        "scoring": {"primary_metric": "ipsae_min", "pose_metric": "sc_dockq"},
        "optimization": {
            "rounds": rounds,
            "early_stop_margin": MEASURED_EARLY_STOP_MARGIN,
            "operators": ["inverse-folding-resample", "point-mutation"],
            "adapter_policies": [
                {
                    "adapter_id": controller.ADAPTER_ID,
                    "operations": ["inverse-folding-resample", "point-mutation"],
                    "parameter_ranges": {
                        "sampling_temperature": {"type": "number", "minimum": 0.05, "maximum": 1.0},
                        "mutation_count": {"type": "integer", "minimum": 1, "maximum": 20},
                    },
                }
            ],
        },
    }


def candidate(root: Path, candidate_id: str, round_number: int) -> dict:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    sequence_path = root / f"{candidate_id}.fasta"
    pose_path = root / f"{candidate_id}.pdb"
    sequence_path.write_text(f">{candidate_id}\n{sequence}\n", encoding="ascii")
    pose_path.write_text(f"POSE {candidate_id}\n", encoding="ascii")
    return {
        "candidate_id": candidate_id,
        "target_id": "target",
        "target_sha256": "t" * 64,
        "parent_candidate_id": None if round_number == 0 else "starting-design",
        "root_candidate_id": "starting-design",
        "origin_generator": "test",
        "root_backbone_id": "starting-design",
        "tm90_cluster_id": "cluster",
        "structure_method": "test",
        "seq_method": "test",
        "fold_class": "test",
        "optimization_round": round_number,
        "sequence_path": str(sequence_path.resolve()),
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "sequence_length": len(sequence),
        "design_pose_path": str(pose_path.resolve()),
        "design_pose_sha256": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
        "status": "generated",
        "promotion_status": "promoted" if round_number == 0 else None,
    }


class LoopTurnTests(unittest.TestCase):
    def test_profiles_declare_only_bound_operations_and_the_measured_margin(self) -> None:
        profile_root = package_root() / "data" / "templates" / "profiles"
        roster_text = (package_root() / "data" / "model-roster.json").read_text(encoding="utf-8")
        self.assertIn('"value": 0.029008', roster_text)
        for path in sorted(profile_root.glob("*.json")):
            profile = lane.load_profile(path)
            optimization = profile.get("campaign_overrides", {}).get("optimization", {})
            self.assertEqual(
                optimization.get("early_stop_margin"),
                MEASURED_EARLY_STOP_MARGIN,
                path.name,
            )
            operators = optimization.get("operators")
            self.assertIsInstance(operators, list, path.name)
            self.assertTrue(operators, path.name)
            self.assertTrue(set(operators).issubset(controller.IMPLEMENTED_OPERATIONS), path.name)
            operations = optimization["adapter_policies"][0]["operations"]
            self.assertEqual(operators, operations, path.name)

        full_profile = lane.load_profile(profile_root / "full-ensemble.template.json")
        optimization = full_profile["campaign_overrides"]["optimization"]
        self.assertEqual(
            controller.choose_operation({"optimization": optimization}, 1)[0],
            "inverse-folding-resample",
        )
        self.assertEqual(
            controller.choose_operation({"optimization": optimization}, 2)[0],
            "point-mutation",
        )

    def test_early_stop_uses_the_measured_margin_and_requires_a_second_tie(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = controller_config(rounds=3)
            write_jsonl(root / "scores" / "screen-score-table.jsonl", [score_row("initial", 0.80)])
            write_jsonl(
                root / "optimization" / "rounds" / "round-1" / "score-table.jsonl",
                [score_row("round-one", 0.80)],
            )
            write_jsonl(
                root / "optimization" / "rounds" / "round-2" / "score-table.jsonl",
                [score_row("round-two", 0.80)],
            )
            with patch.object(controller.lane, "rank_candidate_cohort", side_effect=rank_rows):
                stop_after_one_tie, reason_after_one_tie = controller.early_stop_decision(
                    config,
                    root,
                    2,
                    [score_row("round-one", 0.80)],
                )
                stop_after_two_ties, reason_after_two_ties = controller.early_stop_decision(
                    config,
                    root,
                    3,
                    [score_row("round-two", 0.80)],
                )

            self.assertFalse(stop_after_one_tie)
            self.assertIsNone(reason_after_one_tie)
            self.assertTrue(stop_after_two_ties)
            self.assertIn("two consecutive rounds", str(reason_after_two_ties))
            self.assertIn("0.029008", str(reason_after_two_ties))

            write_jsonl(
                root / "optimization" / "rounds" / "round-1" / "score-table.jsonl",
                [score_row("small-improvement", 0.81)],
            )
            with patch.object(controller.lane, "rank_candidate_cohort", side_effect=rank_rows):
                stop, reason = controller.early_stop_decision(
                    config,
                    root,
                    2,
                    [score_row("small-improvement", 0.81)],
                )

            self.assertTrue(stop)
            self.assertIn("early_stop_margin is 0.029008", str(reason))

    def test_final_rescore_ranks_the_round_one_winner_after_two_rounds(self) -> None:
        config = {
            "profile": {"claim_level": "candidate", "scores_ungated": True},
            "optimization": {"enabled": True, "rounds": 2},
            "targets": [{"target_id": "target", "role": "primary"}],
            "cofold": {"predictors": [{"id": "arm", "enabled": True}]},
            "scoring": {
                "primary_metric": "ipsae_min",
                "pose_metric": "sc_dockq",
                "rank_weights": {"ipsae_min": 1.0, "sc_dockq": 0.0},
                "thresholds": {
                    "minimum_ipsae_min_ensemble": 0.0,
                    "minimum_sc_dockq_ensemble": 0.0,
                    "maximum_clash_count": 10,
                },
            },
            "stages": [
                {
                    "stage_id": "optimization-select-round-2",
                    "outputs": [{"artifact_id": "rescore-candidates"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            starting = candidate(root, "starting-design", 0)
            winner = candidate(root, "round-one-winner", 1)
            later = candidate(root, "round-two-worse", 2)
            artifact_root = root / "artifacts"
            write_json(root / "config.json", config)
            write_json(root / "plan.json", {})
            write_jsonl(artifact_root / "promotion" / "promotion-manifest.jsonl", [starting])
            for round_number, row, score in ((1, winner, 0.95), (2, later, 0.60)):
                round_root = artifact_root / "optimization" / "rounds" / f"round-{round_number}"
                write_jsonl(round_root / "optimized-candidates.jsonl", [row])
                write_jsonl(round_root / "score-table.jsonl", [score_row(row["candidate_id"], score)])
            args = Namespace(
                config=root / "config.json",
                plan=root / "plan.json",
                stage="optimization-select-round-2",
                phase="single",
                attempt_dir=root / "attempt",
                receipts_dir=root / "receipts",
                artifact_root=artifact_root,
            )
            with patch.object(promotion_selector, "load_plan", return_value={}):
                self.assertEqual(promotion_selector.run_stage(args), 0)

            manifest = root / "attempt" / "single" / promotion_selector.OUTPUT_NAME
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [row["candidate_id"] for row in rows],
                ["starting-design", "round-one-winner", "round-two-worse"],
            )
            observations = [
                score_row("starting-design", 0.50),
                score_row("round-one-winner", 0.95),
                score_row("round-two-worse", 0.60),
            ]
            ranked = lane.rank_candidate_cohort(config, observations, [0])
            self.assertEqual(ranked[0]["candidate_id"], "round-one-winner")


if __name__ == "__main__":
    unittest.main()
