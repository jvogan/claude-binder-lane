"""Custom small-campaign seed lists retain complete scoring and control gates."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SKILL_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = SKILL_ROOT / "scripts" / "small_campaign.py"
TEMPLATE = SKILL_ROOT / "scripts" / "small-campaign-settings.template.json"
spec = importlib.util.spec_from_file_location("small_campaign_seed_tests", SCRIPT)
assert spec is not None and spec.loader is not None
small_campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(small_campaign)


class SmallCampaignSeedTests(unittest.TestCase):
    def settings(self, root: Path, seeds: list[int]) -> dict:
        target = root / "target.cif"
        msa = root / "target.a3m"
        target.write_text("target", encoding="utf-8")
        msa.write_text("msa", encoding="utf-8")
        settings = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        settings.update(campaign_id="seed-test", maximum_spend_usd=100.0,
                        maximum_job_estimate_usd=10.0)
        settings["target"].update(sequence="AAAA", structure_path=str(target), msa_path=str(msa))
        settings["design"]["enabled"] = False
        settings["controls"] = {"positive_id": "positive", "negative_ids": ["negative"]}
        settings["rescore"]["seeds"] = seeds
        return settings

    def test_seed_lists_are_configurable_and_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = self.settings(Path(directory), [7])
            settings["rescore"].pop("ranking_rule")
            for seeds in ([7], [2, 7, 11], [0, 1, 2, 3, 4]):
                settings["rescore"]["seeds"] = seeds
                small_campaign._validate_settings(settings)
            for seeds in ([], [7, 7], [-1, 2], [False, 2], [2, 1.5]):
                settings["rescore"]["seeds"] = seeds
                with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "rescore.seeds"):
                    small_campaign._validate_settings(settings)
            settings["rescore"]["seeds"] = [7]
            for rule in (small_campaign.DEFAULT_RANKING_RULE, "mean_of_arm_best_ipsae"):
                settings["rescore"]["ranking_rule"] = rule
                small_campaign._validate_settings(settings)
            settings["rescore"]["ranking_rule"] = "best_of_any_arm"
            with self.assertRaisesRegex(ValueError, "rescore.ranking_rule"):
                small_campaign._validate_settings(settings)

    def test_roster_plan_records_seed_driven_workload_without_changing_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self.settings(root, [7])
            settings["rescore"]["ranking_rule"] = "mean_of_arm_best_ipsae"
            jobs = [{"name": cid, "target": "AAAA", "binder": "CCCC"}
                    for cid in ("positive", "negative", "candidate", "steady")]
            settings_path = root / "settings.json"
            jobs_path = root / "jobs.json"
            out = root / "plan.json"
            small_campaign._write(jobs_path, jobs)
            for seeds in ([7], [2, 7, 11], [0, 1, 2, 3, 4]):
                with self.subTest(seeds=seeds):
                    settings["rescore"]["seeds"] = seeds
                    small_campaign._write(settings_path, settings)
                    args = type("Args", (), {"settings": str(settings_path),
                                             "jobs": str(jobs_path), "out": str(out)})()
                    with redirect_stdout(io.StringIO()):
                        small_campaign.plan(args)
                    plan = small_campaign._json(out)
                    self.assertEqual(plan["rescore_seed_count"], len(seeds))
                    self.assertEqual(plan["rescore_seed_ids"], seeds)
                    self.assertEqual(plan["planned_rescore_jobs"], 2 * len(seeds))
                    self.assertEqual(plan["planned_prediction_calls"], len(jobs) * 2 * len(seeds))
                    self.assertEqual(plan["settings"]["rescore"]["ranking_rule"],
                                     "mean_of_arm_best_ipsae")
                    self.assertEqual(plan["plan_sha256"], small_campaign._digest(
                        {key: value for key, value in plan.items() if key != "plan_sha256"}))

    def test_report_uses_all_declared_seeds_and_keeps_control_gate(self):
        for seeds in ([7], [2, 7, 11]):
            with self.subTest(seeds=seeds), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                settings = self.settings(root, seeds)
                settings["rescore"].pop("ranking_rule")
                small_campaign._validate_settings(settings)
                jobs = [{"name": cid, "target": "AAAA", "binder": "CCCC"}
                        for cid in ("positive", "negative", "candidate", "steady")]
                plan = {"settings": settings, "jobs": jobs, "plan_sha256": "plan-digest"}
                approval = {"campaign_authorization_id": "authorization", "approval_ref": "approval",
                            "maximum_spend_usd": 100.0}
                scores = []
                receipt_records = []
                receipts_by_ref = {}
                for arm in settings["rescore"]["predictors"]:
                    for seed_index, seed in enumerate(seeds):
                        job_ref = f"{arm}-{seed}"
                        job_root = root / job_ref
                        job_root.mkdir()
                        worker_path = job_root / "worker-receipt.json"
                        worker_path.write_text("{}", encoding="utf-8")
                        artifacts = []
                        candidate_values = ([0.4, 0.6, 0.95] if arm == "esmfold2-kit"
                                            else [0.5, 0.65, 0.8]) if len(seeds) == 3 else [0.65]
                        for cid, ipsae in (("positive", 0.8), ("negative", 0.2),
                                           ("candidate", candidate_values[seed_index]), ("steady", 0.7)):
                            complex_path = job_root / f"{cid}.cif"
                            pae_path = job_root / f"{cid}.npz"
                            complex_path.write_text(f"{cid}-{arm}-{seed}", encoding="utf-8")
                            pae_path.write_text("pae", encoding="utf-8")
                            complex_digest = small_campaign._file_digest(complex_path)
                            pae_digest = small_campaign._file_digest(pae_path)
                            artifacts.extend(({"path": complex_path.name, "sha256": complex_digest},
                                              {"path": pae_path.name, "sha256": pae_digest}))
                            scores.append({"candidate_id": cid, "arm": arm, "seed": seed,
                                           "ipsae_min": ipsae, "pose_rmsd": 1.0,
                                           "complex_path": str(complex_path), "complex_sha256": complex_digest,
                                           "pae_path": str(pae_path), "pae_sha256": pae_digest,
                                           "pae_orientation": "aligned_rows", "job_ref": job_ref})
                        receipt_records.append({"job_ref": job_ref})
                        receipts_by_ref[job_ref] = {
                            "job_id": job_ref, "job_ref": job_ref,
                            "worker": {"stage": "rescore", "arm": arm, "seed": seed,
                                       "artifacts": artifacts},
                            "worker_receipt_path": str(worker_path), "wall_seconds": 10.0,
                            "settled_cost_usd": 0.1, "list_rate_estimate_usd": 0.2,
                            "start": datetime(2026, 1, 1, tzinfo=timezone.utc),
                            "end": datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc),
                        }
                scores_path = root / "scores.json"
                receipts_path = root / "receipts.json"
                small_campaign._write(scores_path, scores)
                small_campaign._write(receipts_path, receipt_records)
                args = type("Args", (), {"plan": "unused", "approval": "unused",
                                         "scores": str(scores_path), "receipts": str(receipts_path),
                                         "linked_phase": None, "out": str(root / "report.json")})()
                with patch.object(small_campaign, "_approved", return_value=(plan, approval)), \
                     patch.object(small_campaign, "_validated_provider_receipt",
                                  side_effect=lambda rec, *_: receipts_by_ref[rec["job_ref"]]):
                    missing_scores = scores[:-1]
                    small_campaign._write(scores_path, missing_scores)
                    with self.assertRaisesRegex(ValueError, rf"{len(seeds)}-seed two-arm score matrix incomplete: 1 missing"):
                        small_campaign.report(args)
                    small_campaign._write(scores_path, scores)
                    self.assertEqual(small_campaign.report(args), 0)
                    result = small_campaign._json(args.out)
                    self.assertEqual(result["rescore_seeds"], seeds)
                    self.assertEqual(result["ranking_rule"], small_campaign.DEFAULT_RANKING_RULE)
                    self.assertIn(f"{len(seeds)} seed-labeled prediction calls", result["claim"])
                    self.assertIn(", ".join(map(str, seeds)), result["claim"])
                    self.assertTrue(result["ranking"][0]["passed"])
                    self.assertEqual(result["ranking"][0]["candidate_id"], "steady")
                    default_candidate = next(item for item in result["ranking"] if item["candidate_id"] == "candidate")
                    default_components = [default_candidate["arms"][arm]["control_normalized"]
                                          for arm in settings["rescore"]["predictors"]]
                    self.assertAlmostEqual(default_candidate["rank_score"], sum(default_components) / 2)
                    self.assertEqual(result["cost"]["provider_job_count"], 2 * len(seeds))
                    self.assertAlmostEqual(result["cost"]["settled_usd"], 0.2 * len(seeds))
                    self.assertAlmostEqual(result["cost"]["list_rate_estimate_usd"], 0.4 * len(seeds))
                    settings["rescore"]["ranking_rule"] = "mean_of_arm_best_ipsae"
                    self.assertEqual(small_campaign.report(args), 0)
                    best_result = small_campaign._json(args.out)
                    self.assertEqual(best_result["ranking_rule"], "mean_of_arm_best_ipsae")
                    best_candidate = next(item for item in best_result["ranking"] if item["candidate_id"] == "candidate")
                    best_components = [best_candidate["arms"][arm]["best_ipsae"]
                                       for arm in settings["rescore"]["predictors"]]
                    self.assertAlmostEqual(best_candidate["rank_score"], sum(best_components) / 2)
                    self.assertAlmostEqual(best_candidate["arms"]["esmfold2-kit"]["mean_ipsae"], 0.65)
                    self.assertEqual(best_candidate["passed"], default_candidate["passed"])
                    if len(seeds) == 3:
                        self.assertEqual(best_result["ranking"][0]["candidate_id"], "candidate")
                        self.assertAlmostEqual(best_candidate["rank_score"], 0.875)
                    else:
                        self.assertEqual(best_result["ranking"][0]["candidate_id"], "steady")
                    self.assertEqual(best_result["cost"], result["cost"])
                    with Path(best_result["exports"]["ranked_csv"]).open(newline="", encoding="utf-8") as stream:
                        csv_rows = list(csv.DictReader(stream))
                    self.assertEqual(csv_rows[0]["ranking_rule"], "mean_of_arm_best_ipsae")
                    self.assertIn("esmfold2-kit_best_ipsae", csv_rows[0])
                    self.assertIn("esmfold2-kit_mean_ipsae", csv_rows[0])
                    summary = Path(best_result["exports"]["summary_text"]).read_text(encoding="utf-8")
                    self.assertIn("raw mean of the two per-arm best-of-seeds ipSAE_min values", summary)
                    if len(seeds) == 3:
                        for score in scores:
                            if score["candidate_id"] == "candidate" and score["arm"] == "esmfold2-kit":
                                score["ipsae_min"] = 0.99 if score["seed"] == seeds[-1] else 0.1
                        small_campaign._write(scores_path, scores)
                        small_campaign.report(args)
                        gated = small_campaign._json(args.out)
                        candidate = next(item for item in gated["ranking"] if item["candidate_id"] == "candidate")
                        self.assertFalse(candidate["passed"])
                        self.assertGreater(candidate["arms"]["esmfold2-kit"]["best_ipsae"], 0.9)
                    for score in scores:
                        if score["candidate_id"] == "positive":
                            score["ipsae_min"] = 0.1
                    small_campaign._write(scores_path, scores)
                    with self.assertRaisesRegex(ValueError, "positive control does not separate"):
                        small_campaign.report(args)


if __name__ == "__main__":
    unittest.main()
