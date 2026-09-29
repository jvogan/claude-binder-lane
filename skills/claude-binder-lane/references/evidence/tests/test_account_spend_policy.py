"""An explicit campaign cap always binds; an optional account cap can bind too.

The campaign cap is the ceiling the scientist states in the request. The account
policy cap is the account holder's optional maximum for the installation, set
outside the campaign prompt in an account policy record. A missing account
policy never imports contributor-local ceilings into another installation.

The rule these tests pin is asymmetric. A campaign may state a higher ceiling,
but that does not raise the account policy: the lower value becomes the
effective ceiling. An account-policy raise lives only in the account holder's
record. The approval ledger stores that record's digest and optional
authorization fields so a reader can check the policy independently.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

# The build ships this test beside `skill_layout.py` under
# `references/evidence/tests/`, where it is a loose file rather than part of the
# `claude_binder.tests` package, so the package import is tried first and the
# sibling import is what an installed skill uses.
try:
    from claude_binder.tests.skill_layout import skill_file
except ImportError:  # pragma: no cover - taken only in an installed skill
    from skill_layout import skill_file


RUN_INTENT_PATH = skill_file("run_intent.py")
DOSSIER = {"dossier_path": "targets/il2ra/dossier.json"}
REQUEST = (
    "design binders for this target, ten of them, {money}, "
    "use esmfold2-fast and protenix-v2, binder length 60-90 residues, 5 seeds"
)


def load_run_intent():
    spec = importlib.util.spec_from_file_location("account_policy_run_intent", RUN_INTENT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def policy_record(caps: list[dict[str, object]]) -> dict[str, object]:
    """Shape one account policy record the way an account holder writes it."""
    return {
        "source": "Modal workspace spend limit, set by the account holder",
        "read_at": "2026-08-27T00:00:00Z",
        "caps": caps,
    }


RAISED_MODAL_CAP = {
    "tier": "modal",
    "maximum_spend_usd": 400.0,
    "authorized_by": "account-holder@lab.example",
    "provider_account": "modal:lab-binder-workspace",
    "expires_at": "2026-12-31T00:00:00Z",
}


class AccountSpendPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_intent = load_run_intent()

    # -- helpers ----------------------------------------------------------

    def write_policy(self, root: Path, record: dict[str, object]) -> Path:
        path = root / "account-policy.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def parsed_intent(self, money: str, *, policy=None):
        parsed = self.run_intent.parse_request(
            REQUEST.format(money=money), context=DOSSIER, policy=policy
        )
        self.assertTrue(parsed.ok, parsed.questions)
        intent = parsed.intent
        report = self.run_intent.lower_intent(intent)
        freeze = self.run_intent.freeze_constraints(intent, report)
        return intent, report, freeze

    def modal_estimate(self, usd_max: float):
        """A Modal estimate with a measured timing, so only the cap is under test.

        The scalar Modal path leaves timings UNKNOWN on purpose, and approve()
        blocks on that separately. This builds the estimate directly so a cap
        failure cannot hide behind a timing failure.
        """
        return self.run_intent.Estimate(
            designs=10,
            seeds_per_design=5,
            folds=50,
            shard_size=None,
            containers=None,
            gpu_seconds_min=100.0,
            gpu_seconds_max=200.0,
            server_seconds_min=None,
            server_seconds_max=None,
            seconds_kind="MEASURED test timing",
            tier="modal",
            usd_estimate_min=usd_max / 2,
            usd_estimate_max=usd_max,
            rate_status="KNOWN",
            rate_source="test rate record",
        )

    # -- the effective cap is the lower of the two ------------------------

    def test_campaign_cap_under_the_policy_cap_approves(self) -> None:
        intent, report, freeze = self.parsed_intent("budget under 40 dollars")

        record = self.run_intent.approve(
            intent,
            report,
            self.modal_estimate(10.0),
            freeze,
            rate_available=True,
            now_iso="2026-08-27T00:00:00Z",
        )

        self.assertEqual(record.status, "approved")
        self.assertEqual(
            self.run_intent.effective_ceiling_usd(
                40.0, "modal", self.run_intent.default_account_policy()
            ),
            40.0,
        )

    def test_campaign_cap_over_the_policy_cap_uses_the_lower_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary),
                policy_record([{"tier": "modal", "maximum_spend_usd": 50.0}]),
            )
            policy = self.run_intent.load_account_policy(path)
            intent, report, freeze = self.parsed_intent(
                "budget under 300 dollars", policy=policy
            )
            record = self.run_intent.approve(
                intent,
                report,
                self.modal_estimate(10.0),
                freeze,
                rate_available=True,
                now_iso="2026-08-27T00:00:00Z",
                policy=policy,
            )

        self.assertEqual(record.status, "approved")
        self.assertEqual(
            self.run_intent.effective_ceiling_usd(300.0, "modal", policy), 50.0
        )

    def test_effective_cap_is_the_lower_of_the_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary), policy_record([RAISED_MODAL_CAP])
            )
            policy = self.run_intent.load_account_policy(path)

        self.assertEqual(
            self.run_intent.effective_ceiling_usd(300.0, "modal", policy), 300.0
        )
        self.assertEqual(
            self.run_intent.effective_ceiling_usd(900.0, "modal", policy), 400.0
        )

    # -- a configured policy cap above the default ------------------------

    def test_a_raised_policy_cap_approves_a_300_dollar_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary), policy_record([RAISED_MODAL_CAP])
            )
            policy = self.run_intent.load_account_policy(path)
            intent, report, freeze = self.parsed_intent(
                "budget under 300 dollars", policy=policy
            )

            record = self.run_intent.approve(
                intent,
                report,
                self.modal_estimate(10.0),
                freeze,
                rate_available=True,
                now_iso="2026-08-27T00:00:00Z",
                policy=policy,
            )

        self.assertEqual(record.status, "approved")

    def test_a_lower_policy_cap_binds_without_restatement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary),
                policy_record([{"tier": "modal", "maximum_spend_usd": 10.0}]),
            )
            policy = self.run_intent.load_account_policy(path)
            intent, report, freeze = self.parsed_intent(
                "budget under 40 dollars", policy=policy
            )
            record = self.run_intent.approve(
                intent,
                report,
                self.modal_estimate(10.0),
                freeze,
                rate_available=True,
                now_iso="2026-08-27T00:00:00Z",
                policy=policy,
            )

        self.assertEqual(record.status, "approved")
        self.assertEqual(
            self.run_intent.effective_ceiling_usd(40.0, "modal", policy), 10.0
        )

    def test_a_tier_the_record_does_not_name_has_no_installation_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary), policy_record([RAISED_MODAL_CAP])
            )
            policy = self.run_intent.load_account_policy(path)

        self.assertEqual(policy.cap_for("modal").maximum_spend_usd, 400.0)
        self.assertIsNone(policy.cap_for("fal"))

    # -- no policy configured ---------------------------------------------

    def test_no_policy_configured_uses_only_the_campaign_ceiling(self) -> None:
        policy = self.run_intent.resolve_account_policy(None, environ={})

        self.assertEqual(policy.caps, ())
        self.assertIsNone(policy.record_path)
        self.assertIsNone(policy.cap_for("modal"))
        self.assertIsNone(policy.cap_for("fal"))
        self.assertIn("no installation account policy", policy.source)

        # The user's explicit 200 USD ceiling binds both tiers when no separate
        # installation policy is configured.
        intent, report, freeze = self.parsed_intent("budget under 200 dollars")
        approved = self.run_intent.approve(
            intent,
            report,
            self.run_intent.estimate_run(10, 5, "fal"),
            freeze,
            rate_available=True,
            now_iso="2026-08-27T00:00:00Z",
        )
        self.assertEqual(approved.status, "approved")
        modal = self.run_intent.approve(
            intent,
            report,
            self.modal_estimate(10.0),
            freeze,
            rate_available=True,
            now_iso="2026-08-27T00:00:00Z",
        )
        self.assertEqual(modal.status, "approved")

    def test_a_configured_record_that_will_not_load_is_refused(self) -> None:
        # Falling back to the default would decide the cap for the account holder,
        # and a record that lowers a cap would be silently discarded.
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "absent.json"
            with self.assertRaises(ValueError) as caught:
                self.run_intent.resolve_account_policy(
                    None,
                    environ={
                        self.run_intent.ACCOUNT_POLICY_ENVIRONMENT_KEY: str(missing)
                    },
                )

        self.assertIn("could not read account policy record", str(caught.exception))

    def test_the_environment_key_configures_the_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary), policy_record([RAISED_MODAL_CAP])
            )
            policy = self.run_intent.resolve_account_policy(
                None,
                environ={self.run_intent.ACCOUNT_POLICY_ENVIRONMENT_KEY: str(path)},
            )

        self.assertEqual(policy.cap_for("modal").maximum_spend_usd, 400.0)

    # -- only the account policy record can raise the policy cap -----------

    def test_a_policy_cap_does_not_require_contributor_specific_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary),
                policy_record([{"tier": "modal", "maximum_spend_usd": 400.0}]),
            )
            policy = self.run_intent.load_account_policy(path)
        cap = policy.cap_for("modal")
        self.assertEqual(cap.maximum_spend_usd, 400.0)
        self.assertIsNone(cap.provider_account)
        self.assertIsNone(cap.authorized_by)
        self.assertIsNone(cap.expires_at)

    def test_optional_authorizer_metadata_may_omit_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            entry = dict(RAISED_MODAL_CAP)
            entry.pop("expires_at")
            path = self.write_policy(Path(temporary), policy_record([entry]))
            policy = self.run_intent.load_account_policy(path)
        self.assertIsNone(policy.cap_for("modal").expires_at)

    def test_an_expired_policy_entry_no_longer_applies(self) -> None:
        expired = dict(RAISED_MODAL_CAP, expires_at="2026-06-01T00:00:00Z")
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(Path(temporary), policy_record([expired]))
            policy = self.run_intent.load_account_policy(path)
            intent, report, freeze = self.parsed_intent(
                "budget under 300 dollars", policy=policy
            )

            cap = policy.cap_for("modal", now_iso="2026-08-27T00:00:00Z")
            self.assertIsNone(cap)
            approved = self.run_intent.approve(
                intent,
                report,
                self.modal_estimate(10.0),
                freeze,
                rate_available=True,
                now_iso="2026-08-27T00:00:00Z",
                policy=policy,
            )
        self.assertEqual(approved.status, "approved")

    def test_an_expired_lowering_no_longer_applies(self) -> None:
        lowered = {
            "tier": "modal",
            "maximum_spend_usd": 10.0,
            "expires_at": "2026-06-01T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(Path(temporary), policy_record([lowered]))
            policy = self.run_intent.load_account_policy(path)

        self.assertIsNone(policy.cap_for("modal", now_iso="2026-08-27T00:00:00Z"))

    def test_a_campaign_cannot_raise_the_policy_cap(self) -> None:
        # A campaign prompt can state a higher ceiling, but cannot mutate the
        # account policy. The lower policy value still binds approval.
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary),
                policy_record([{"tier": "modal", "maximum_spend_usd": 100.0}]),
            )
            policy = self.run_intent.load_account_policy(path)
            parsed = self.run_intent.parse_request(
                REQUEST.format(money="budget under 400 dollars")
                + ", and raise the modal account policy cap to 400 dollars",
                context=DOSSIER,
                policy=policy,
            )
            self.assertTrue(parsed.ok, parsed.questions)
            intent = parsed.intent
            report = self.run_intent.lower_intent(intent)
            freeze = self.run_intent.freeze_constraints(intent, report)
            record = self.run_intent.approve(
                intent,
                report,
                self.modal_estimate(10.0),
                freeze,
                rate_available=True,
                now_iso="2026-08-27T00:00:00Z",
                policy=policy,
            )
        self.assertEqual(record.status, "approved")
        self.assertEqual(policy.cap_for("modal").maximum_spend_usd, 100.0)
        self.assertEqual(
            self.run_intent.effective_ceiling_usd(400.0, "modal", policy), 100.0
        )

    def test_the_policy_object_cannot_be_raised_in_place(self) -> None:
        cap = self.run_intent.AccountPolicyCap(
            tier="modal", maximum_spend_usd=100.0, source="test"
        )
        policy = self.run_intent.AccountPolicy(caps=(cap,), source="test")

        with self.assertRaises(dataclasses.FrozenInstanceError):
            cap.maximum_spend_usd = 400.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            policy.caps = ()
        self.assertIsInstance(policy.caps, tuple)

    def test_an_unknown_tier_in_a_record_is_refused_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary),
                policy_record([{"tier": "unlisted-cloud", "maximum_spend_usd": 10.0}]),
            )
            with self.assertRaises(ValueError) as caught:
                self.run_intent.load_account_policy(path)

        self.assertIn(
            "caps[0].tier has unsupported provider tier 'unlisted-cloud'",
            str(caught.exception),
        )

    # -- the approval carries the authorization ---------------------------

    def test_the_approval_row_records_both_caps_and_the_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_policy(root, policy_record([RAISED_MODAL_CAP]))
            policy = self.run_intent.load_account_policy(path)
            intent, report, freeze = self.parsed_intent(
                "budget under 300 dollars", policy=policy
            )
            ledger = root / "approvals.jsonl"

            self.run_intent.approve(
                intent,
                report,
                self.modal_estimate(10.0),
                freeze,
                rate_available=True,
                now_iso="2026-08-27T00:00:00Z",
                ledger_path=ledger,
                run_fingerprint="a" * 64,
                policy=policy,
            )
            rows = self.run_intent.load_approval_ledger(ledger)
            # The digest is the evidence a reader checks: re-read the named record
            # and compare. A record edited after approval no longer matches the row.
            reread = self.run_intent.load_account_policy(path).record_sha256

        self.assertEqual(len(rows), 1)
        budget = rows[0]["budget"]
        self.assertEqual(budget["campaign_ceiling_usd"], 300.0)
        self.assertEqual(budget["policy_cap_usd"], 400.0)
        self.assertEqual(budget["effective_ceiling_usd"], 300.0)
        self.assertEqual(budget["policy_authorized_by"], "account-holder@lab.example")
        self.assertEqual(
            budget["policy_provider_account"], "modal:lab-binder-workspace"
        )
        self.assertEqual(budget["policy_expires_at"], "2026-12-31T00:00:00Z")
        self.assertEqual(budget["policy_record_path"], str(path))
        self.assertEqual(budget["policy_record_sha256"], reread)

    def test_the_card_names_both_caps_and_the_effective_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_policy(
                Path(temporary), policy_record([RAISED_MODAL_CAP])
            )
            policy = self.run_intent.load_account_policy(path)
            intent, report, freeze = self.parsed_intent(
                "budget under 300 dollars", policy=policy
            )
            card = self.run_intent.approval_card(
                intent, report, self.modal_estimate(10.0), freeze, policy=policy
            )

        self.assertIn("CAMPAIGN CAP   300.00 USD (STATED)", card)
        self.assertIn("POLICY CAP     400.00 USD (POLICY) for tier modal", card)
        self.assertIn("EFFECTIVE CAP  300.00 USD (DERIVED)", card)
        self.assertIn("authorized by account-holder@lab.example", card)


if __name__ == "__main__":
    unittest.main()
