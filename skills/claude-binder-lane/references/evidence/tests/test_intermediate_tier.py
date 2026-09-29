"""The full ensemble narrows SCREEN before configured-seed parent selection."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from claude_binder import lane
from claude_binder.adapters import esmfold2_predictor, screen_survivor_selector
from claude_binder.paths import package_root


TEMPLATES = package_root() / "data/templates"


def configured_profile(profile_name: str = "full-ensemble.template.json") -> dict:
    config = lane.load_json(TEMPLATES / "campaign.template.json")
    profile = lane.load_profile(TEMPLATES / "profiles" / profile_name)
    for section, overrides in profile.pop("campaign_overrides", {}).items():
        config[section] = lane.deep_merge_dict(config.get(section, {}), overrides)
    config.update({key: value for key, value in profile.items() if key not in {"schema_version", "template_id"}})
    if profile_name == "published-baseline-fidelity.template.json":
        config.setdefault("provider_endpoints", {}).update({
            key: "https://example.invalid/fixture"
            for key in ("boltzgen_fal_url", "freebindcraft_fal_url",
                        "proteina_complexa_fal_url", "pxdesign_fal_url")
        })
    lane.bind_dynamic_stage_contracts(config)
    return config


class IntermediateTierTests(unittest.TestCase):
    def test_full_and_published_graphs_funnel_before_each_parent_selection(self) -> None:
        for profile_name in ("full-ensemble.template.json", "published-baseline-fidelity.template.json"):
            with self.subTest(profile=profile_name):
                config = configured_profile(profile_name)
                self.assertTrue(lane.validate_campaign(config, allow_placeholders=True)["ok"])
                stage_map = {stage["stage_id"]: stage for stage in config["stages"]}
                self.assertEqual(stage_map["screen-survivors"]["inputs"][0], "score-screen:screen-score-table")
                self.assertEqual(stage_map["cofold-intermediate-esmfold2"]["inputs"][0], "screen-survivors:intermediate-candidates")
                self.assertEqual(stage_map["cofold-intermediate-esmfold2"]["fanout"]["count_from"]["stage_id"], "screen-survivors")
                self.assertEqual(stage_map["promote"]["inputs"][0], "score-intermediate:intermediate-score-table")
                self.assertEqual(stage_map["optimization-plan"]["inputs"][1], "score-intermediate:intermediate-score-table")
                self.assertIn("score-intermediate", stage_map["optimization-plan"]["depends_on"])
                self.assertEqual(stage_map["cofold-screen-esmfold2"]["outputs"][0]["records_per_count"], 1)
                self.assertEqual(stage_map["cofold-intermediate-esmfold2"]["outputs"][0]["records_per_count"], 5)
                estimate = lane.estimate_fanout(config)["counts"]
                self.assertEqual(estimate["intermediate_survivors_upper_bound"],
                                 max(config["optimization"]["parent_count_per_round"],
                                     math.ceil(estimate["generated_candidates_upper_bound"] * 0.2)))
                lane.expand_optimization_rounds(config)
                expanded = {stage["stage_id"]: stage for stage in config["stages"]}
                self.assertEqual(expanded["optimization-plan-round-1"]["inputs"][1], "score-intermediate:intermediate-score-table")
                self.assertEqual(expanded["optimization-plan-round-2"]["inputs"][1], "optimization-measure-round-1:optimization-score-table")
                self.assertEqual(expanded["optimization-cofold-round-1-esmfold2"]["outputs"][0]["records_per_count"], 5)
                self.assertEqual(esmfold2_predictor.campaign_phase("optimization-cofold-round-1-esmfold2"), "optimization")
                self.assertTrue(lane.validate_campaign(config, allow_placeholders=True)["ok"])

    def test_custom_seed_count_changes_intermediate_and_round_contracts(self) -> None:
        config = configured_profile()
        config["cofold"]["rescore_seeds"] = [4, 7, 9]
        lane.bind_dynamic_stage_contracts(config)
        stage_map = {stage["stage_id"]: stage for stage in config["stages"]}
        self.assertEqual(stage_map["cofold-intermediate-esmfold2"]["outputs"][0]["records_per_count"], 3)
        self.assertEqual(esmfold2_predictor.seeds_for(config, "intermediate"), [4, 7, 9])
        lane.expand_optimization_rounds(config)
        expanded = {stage["stage_id"]: stage for stage in config["stages"]}
        self.assertEqual(expanded["optimization-cofold-round-1-esmfold2"]["outputs"][0]["records_per_count"], 3)
        self.assertEqual(esmfold2_predictor.seeds_for(config, "optimization"), [4, 7, 9])

    def test_small_run_overlay_keeps_its_existing_graph(self) -> None:
        config = configured_profile("small-run.template.json")
        self.assertFalse(lane.intermediate_enabled(config))
        self.assertNotIn("screen-survivors", {stage["stage_id"] for stage in config["stages"]})

    def test_local_fixture_uses_the_real_survivor_selector(self) -> None:
        config = configured_profile("local-contract-test.json")
        selector = next(item for item in config["adapters"]
                        if item["adapter_id"] == "screen-survivor-selector")
        for field in ("toolcheck_argv", "command_argv_template", "parser_argv_template"):
            self.assertEqual(selector[field][2], "claude_binder.adapters.screen_survivor_selector")

    def test_intermediate_rejects_missing_seed_for_a_survivor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scores").mkdir()
            (root / "filters").mkdir()
            survivor = {"candidate_id": "c0", "origin_generator": "g", "sequence_sha256": "s",
                        "design_pose_sha256": "p"}
            for path in (root / "scores/intermediate-candidates.jsonl", root / "filters/passing-candidates.jsonl"):
                path.write_text(json.dumps(survivor) + "\n")
            score_path = root / "scores/intermediate-score-table.jsonl"
            score_path.write_text(json.dumps({
                "target_id": "t", "target_sha256": "h", "candidate_id": "c0",
                "origin_generator": "g", "predictor": "a", "seed": 4,
                "sequence_sha256": "s", "design_pose_sha256": "p",
                "model_revision": "m", "phase": "intermediate", "filter_pass": True,
                "status": "scored",
            }) + "\n")
            config = {
                "cofold": {"intermediate_enabled": True, "intermediate_fraction": 1.0,
                           "rescore_seeds": [4, 7],
                           "predictors": [{"id": "a", "adapter_id": "predictor", "enabled": True}]},
                "optimization": {"parent_count_per_round": 1},
                "targets": [{"target_id": "t", "structure_sha256": "h"}],
                "adapters": [{"adapter_id": "predictor", "model_revision": "m"}],
            }
            with patch.object(lane, "validate_screen_scored_pool", return_value={"ok": True, "errors": []}), \
                 patch.object(lane, "validate_observations", return_value=[]):
                check = lane.validate_intermediate_scored_pool(config, score_path, root)
            self.assertFalse(check["ok"])
            self.assertTrue(any("score keys do not exactly match" in error for error in check["errors"]))

    def test_screen_survivors_are_bounded_and_ranked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cofold": {"screen_seeds": [0], "intermediate_fraction": 0.2},
                      "optimization": {"parent_count_per_round": 2},
                      "generation": {"minimum_generators": 1},
                      "selection": {"minimum_generators": 1, "maximum_fraction_per_generator": 1.0}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            scores = root / "screen.jsonl"
            passing = root / "passing.jsonl"
            scores.write_text("{}\n")
            passing.write_text("".join(json.dumps({"candidate_id": f"c{i}"}) + "\n" for i in range(10)))
            args = Namespace(config=config_path, plan=root / "plan.json", receipts_dir=root,
                             stage="screen-survivors", artifact_root=root, attempt_dir=root / "attempt",
                             phase="single", count=1)
            ranked = [{"candidate_id": f"c{i}", "generator": "g", "eligible": True} for i in range(10)]
            with patch.object(screen_survivor_selector, "load_plan", return_value={}), \
                 patch.object(screen_survivor_selector, "input_files", side_effect=[({}, [scores]), ({}, [passing])]), \
                 patch.object(lane, "validate_screen_scored_pool", return_value={"ok": True}), \
                 patch.object(lane, "validate_control_calibration", return_value={"ok": True}), \
                 patch.object(lane, "rank_candidate_cohort", return_value=ranked), \
                 patch.object(lane, "apply_declared_ranking_mode", side_effect=lambda _config, rows, **_kwargs: rows), \
                 patch.object(lane, "_rank_sort_key", side_effect=lambda row, _config: int(row["candidate_id"][1:])):
                self.assertEqual(screen_survivor_selector.run(args), 0)
            selected = lane.load_jsonl(screen_survivor_selector.output_path(args))
            self.assertEqual([row["candidate_id"] for row in selected], ["c0", "c1"])
            self.assertTrue(all(row["screen_survivor_selected_count"] == 2 for row in selected))

    def test_screen_survivors_reserve_diverse_parent_slate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"cofold": {"screen_seeds": [0], "intermediate_fraction": 0.2},
                      "optimization": {"parent_count_per_round": 3},
                      "generation": {"minimum_generators": 2},
                      "selection": {"minimum_generators": 3, "maximum_fraction_per_generator": 0.5}}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            scores = root / "screen.jsonl"
            passing = root / "passing.jsonl"
            scores.write_text("{}\n")
            passing.write_text("".join(json.dumps({"candidate_id": f"c{i}"}) + "\n" for i in range(4)))
            args = Namespace(config=config_path, plan=root / "plan.json", receipts_dir=root,
                             stage="screen-survivors", artifact_root=root, attempt_dir=root / "attempt",
                             phase="single", count=1)
            ranked = [{"candidate_id": f"c{i}", "generator": generator, "eligible": True}
                      for i, generator in enumerate(("g1", "g1", "g2", "g3"))]
            with patch.object(screen_survivor_selector, "load_plan", return_value={}), \
                 patch.object(screen_survivor_selector, "input_files", side_effect=[({}, [scores]), ({}, [passing])]), \
                 patch.object(lane, "validate_screen_scored_pool", return_value={"ok": True}), \
                 patch.object(lane, "validate_control_calibration", return_value={"ok": True}), \
                 patch.object(lane, "rank_candidate_cohort", return_value=ranked), \
                 patch.object(lane, "apply_declared_ranking_mode", side_effect=lambda _config, rows, **_kwargs: rows), \
                 patch.object(lane, "_rank_sort_key", side_effect=lambda row, _config: int(row["candidate_id"][1:])):
                self.assertEqual(screen_survivor_selector.run(args), 0)
            selected = lane.load_jsonl(screen_survivor_selector.output_path(args))
            self.assertEqual([row["candidate_id"] for row in selected], ["c0", "c2", "c3"])


if __name__ == "__main__":
    unittest.main()
