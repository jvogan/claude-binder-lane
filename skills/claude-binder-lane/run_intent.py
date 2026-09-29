#!/usr/bin/env python3
"""Plain-language entry point for the Claude binder lane.

A scientist says something like "design binders for this target, n=10, under
200 dollars". This module turns that sentence into three things:

1. A typed RunIntent that holds exactly what was said and nothing more.
2. A lowering of that intent into the campaign config the executor in
   lane.py accepts, plus an honest list of what it cannot fill.
3. A Constraint Freeze digest over the intent and resolved settings, so a
   run can be identified and re-run.

It also renders the approval card the user sees before local executor dispatch,
and gates approval on the estimate and the ceiling.

Hard rules this module obeys:

- It never invents a price, a threshold, or a timing. Every constant here
  is copied from a file, and the comment names the file. Anything without
  a source is carried as UNKNOWN.
- It refuses ambiguity instead of guessing. A wrong N costs real money.
- Standard library only. It sits beside lane.py, which is stdlib only.
- Output is candidate-level computational prediction. It is never evidence
  of binding or function.

Files this module cites, at the paths an install has:

- Executor and validator:
  claude_binder/lane.py
- Campaign config shape:
  claude_binder/data/templates/campaign.template.json
- Profile with predictor modes and seed counts:
  claude_binder/data/templates/profiles/small-run.template.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shlex
import sys
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Constants that were read from files, each with its source.
# ---------------------------------------------------------------------------

# Measured billed seconds per unit, read 2026-08-23.
# The billed values include setup and drain, which fal charges. They are the
# correct units for an approval ceiling.
RFDIFFUSION3_BILLED_SECONDS_PER_BACKBONE = 39.9
PROTEINMPNN_BILLED_SECONDS_PER_DESIGN_CALL = 69.8
ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MIN = 214.6
ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MAX = 337.1
# Modal timeouts are optional knobs that must be read from config at run
# time. The values 3600 and 2700 appear only as platform examples, so no
# Modal constant lives here.
MODAL_SHARD_SIZE = None

# Supported account-policy tiers.  No spending limit ships as a default.  When
# no account-policy record is configured, the explicit campaign ceiling is the
# only dollar ceiling.  An installation may add a separate account cap through
# the policy record below.
STANDING_CEILING_USD: dict[str, Optional[float]] = {
    "modal": None,
    "fal": None,
    "runpod": None,
    "lambda": None,
}

# Custom campaigns accept any positive ordered binder range and any nonempty
# distinct rescore seed set. Published baseline fidelity keeps its five-seed
# policy. Selected tools may impose tighter limits during their own preflight.
MIN_CUSTOM_RESCORE_SEEDS = 1
MIN_BASELINE_RESCORE_SEEDS = 5

# Predictor modes enabled in the deployed small-run profile
# (small-run.template.json, cofold.predictors). The loader below reads this
# list from the template when the package is present. The fallback is the same
# list for the case where the template file is absent, so it changes whenever
# the profile's enabled predictor set changes.
KNOWN_PREDICTOR_ARMS_FALLBACK = ("esmfold2-fast",)

# Sources cited in output so the user can check every number themselves.
SOURCE_EXECUTOR = "claude_binder/lane.py"
SOURCE_TIMINGS = "measured fal billed seconds per unit, including setup and drain, read 2026-08-23"
SOURCE_PRICE_WARNING = (
    "the provider billing page verifies the rate, and no CLI reports it. A rate "
    "record is read at run time or the price remains UNKNOWN."
)

# These adapters have no provider call. The resolved bundle records both as
# standard-library Python commands. runtime_validator.py states that it only
# performs local checks. novelty_filter.py reads declared local metric sources.
LOCAL_NO_PROVIDER_ADAPTER_IDS = frozenset({"runtime-validator", "novelty-filter"})
PROVIDER_FACING_ADAPTER_ROLES = frozenset(
    {
        "backbone-generator",
        "codesign-generator",
        "sequence-designer",
        "cofold-predictor",
        "control-builder",
    }
)
PAID_STAGE_PROVIDER_SCHEMA_VERSION = 1

# Do NOT add a rule here that reads an adapter's own interpreter. control_builder
# runs standard-library Python and shells out to a predictor adapter through
# subprocess, so its interpreter says nothing about whether the stage spends.
# A rule keyed on environment_identity would let a folding stage past the
# approval ceiling. A stage without an explicit free declaration stays in the
# paid set.


# ---------------------------------------------------------------------------
# Account spend policy. Two caps bound a run, and the lower one applies.
# ---------------------------------------------------------------------------
#
# The campaign cap is the ceiling the scientist states in the request. It is
# mandatory, it is positive, and it lives on the intent as spend_ceiling.
#
# The account policy cap is the account holder's maximum for this installation.
# It is set outside the campaign prompt, in an account policy record, so an
# agent composing a campaign cannot reach it.  A missing record means no
# installation-wide cap is configured, and no limit is inherited from anywhere
# else.  The approval ledger stores any configured record's digest.

ACCOUNT_POLICY_ENVIRONMENT_KEY = "CLAUDE_BINDER_ACCOUNT_POLICY"
DEFAULT_POLICY_SOURCE = "no installation account policy is configured"


def _utc_instant(value: str) -> datetime:
    """Parse one ISO-8601 timestamp that carries a UTC offset."""
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("the timestamp must carry a UTC offset")
    return instant.astimezone(timezone.utc)


def _clean_text(value: Any) -> Optional[str]:
    """Return a trimmed non-empty string, or None when the value is not one."""
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


@dataclass(frozen=True)
class AccountPolicyCap:
    """One provider tier's account maximum and the person who authorized it."""

    tier: str
    maximum_spend_usd: float
    source: str
    authorized_by: Optional[str] = None
    provider_account: Optional[str] = None
    expires_at: Optional[str] = None

    def is_expired(self, now_iso: str) -> bool:
        """Return whether the authorization has lapsed by the approval time.

        A cap with no expiry never lapses. A cap whose expiry or approval time
        will not parse is treated as lapsed, because an unreadable date is not
        an authorization.
        """
        if self.expires_at is None:
            return False
        try:
            return _utc_instant(self.expires_at) <= _utc_instant(now_iso)
        except ValueError:
            return True

    def authorization_text(self) -> str:
        """Describe who authorized this cap, for a card or a ledger row."""
        if self.authorized_by is None:
            return "configured account policy cap; no optional authorizer metadata recorded"
        return (
            f"authorized by {self.authorized_by} for {self.provider_account} "
            f"until {self.expires_at}"
        )


@dataclass(frozen=True)
class AccountPolicy:
    """The account holder's spend caps for this installation."""

    caps: tuple[AccountPolicyCap, ...] = ()
    record_path: Optional[str] = None
    record_sha256: Optional[str] = None
    read_at: Optional[str] = None
    source: str = DEFAULT_POLICY_SOURCE

    def cap_for(
        self,
        tier: str,
        *,
        now_iso: Optional[str] = None,
    ) -> Optional[AccountPolicyCap]:
        """Return the cap that binds one tier, or None when the tier has none.

        A tier the record does not name has no installation cap.  An expired
        entry no longer applies; the campaign's explicit ceiling still binds.
        """
        recorded = next((cap for cap in self.caps if cap.tier == tier), None)
        if recorded is None:
            return None
        if now_iso is None or not recorded.is_expired(now_iso):
            return recorded
        return None

    def maximum_cap_usd(self) -> Optional[float]:
        """Return the largest cap any tier allows, for a check with no tier yet."""
        amounts = [
            cap.maximum_spend_usd
            for cap in (self.cap_for(tier) for tier in STANDING_CEILING_USD)
            if cap is not None
        ]
        return max(amounts) if amounts else None

    def cap_summary(self) -> str:
        """Render every tier cap for a refusal that has no tier yet."""
        parts = []
        for tier in sorted(STANDING_CEILING_USD):
            cap = self.cap_for(tier)
            if cap is not None:
                parts.append(f"{tier} {cap.maximum_spend_usd:.0f} USD")
        return ", ".join(parts) if parts else "none configured"

    def as_dict(self) -> dict[str, Any]:
        """Return the policy identity that an approval row records."""
        return {
            "source": self.source,
            "record_path": self.record_path,
            "record_sha256": self.record_sha256,
            "read_at": self.read_at,
        }


def default_account_policy() -> AccountPolicy:
    """Return the policy that applies when the account holder configured none."""
    return AccountPolicy()


def load_account_policy(path: Path) -> AccountPolicy:
    """Load the account holder's spend caps from one account policy record.

    The record is explicit input, like a rate record. A caller that supplies
    none uses only the campaign ceiling and inherits no policy from elsewhere.
    Optional provider, authorizer and expiry fields remain provenance.
    """
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read account policy record {path}: {exc}") from exc
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read account policy record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("account policy record must be a JSON object")
    source = _clean_text(value.get("source"))
    read_at = _clean_text(value.get("read_at"))
    entries = value.get("caps")
    if source is None:
        raise ValueError("account policy record source must be a non-empty string")
    if read_at is None:
        raise ValueError(
            "account policy record read_at must be a non-empty timestamp string"
        )
    if not isinstance(entries, list) or not entries:
        raise ValueError("account policy record caps must be a non-empty list")
    caps: list[AccountPolicyCap] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"account policy record caps[{index}] must be an object")
        tier = entry.get("tier")
        if not isinstance(tier, str) or tier not in STANDING_CEILING_USD:
            raise ValueError(
                f"account policy record caps[{index}].tier has unsupported provider tier {tier!r}"
            )
        if tier in seen:
            raise ValueError(f"account policy record repeats tier: {tier}")
        seen.add(tier)
        amount = _finite_bound(entry.get("maximum_spend_usd"))
        if amount is None or amount <= 0:
            raise ValueError(
                f"account policy record caps[{index}].maximum_spend_usd must be positive"
            )
        authorized_by = _clean_text(entry.get("authorized_by"))
        provider_account = _clean_text(entry.get("provider_account"))
        expires_at = _clean_text(entry.get("expires_at"))
        if expires_at is not None:
            try:
                _utc_instant(expires_at)
            except ValueError as exc:
                raise ValueError(
                    f"account policy record caps[{index}].expires_at must be an "
                    f"ISO-8601 timestamp with a UTC offset: {exc}"
                ) from exc
        caps.append(
            AccountPolicyCap(
                tier=tier,
                maximum_spend_usd=amount,
                source=f"the account policy record at {path} ({source})",
                authorized_by=authorized_by,
                provider_account=provider_account,
                expires_at=expires_at,
            )
        )
    return AccountPolicy(
        caps=tuple(caps),
        record_path=str(path),
        record_sha256=hashlib.sha256(payload).hexdigest(),
        read_at=read_at,
        source=f"the account policy record at {path}",
    )


def resolve_account_policy(
    explicit_path: Optional[Path] = None,
    *,
    environ: Optional[dict[str, str]] = None,
) -> AccountPolicy:
    """Return this installation's policy, or no installation cap.

    A configured record that will not load is refused rather than replaced by
    the default. A broken record must not decide a cap in either direction.
    """
    source = os.environ if environ is None else environ
    path = explicit_path
    if path is None:
        configured = str(source.get(ACCOUNT_POLICY_ENVIRONMENT_KEY, "") or "").strip()
        if configured:
            path = Path(configured)
    if path is None:
        return default_account_policy()
    return load_account_policy(path)


def effective_ceiling_usd(
    campaign_ceiling_usd: float,
    tier: str,
    policy: AccountPolicy,
    *,
    now_iso: Optional[str] = None,
) -> float:
    """Return the lower of the campaign ceiling and the account policy cap."""
    cap = policy.cap_for(tier, now_iso=now_iso)
    if cap is None:
        return campaign_ceiling_usd
    return min(campaign_ceiling_usd, cap.maximum_spend_usd)


def _template_dir() -> Optional[Path]:
    """Return an existing template directory, or None when none is reachable."""
    sibling_templates = (
        Path(sys._getframe().f_code.co_filename).resolve().parent / "data" / "templates"
    )
    if sibling_templates.is_dir():
        return sibling_templates

    try:
        from claude_binder.paths import schema_file
    except ImportError:
        return None

    try:
        profile = schema_file("templates/profiles/small-run.template.json")
    except (OSError, ValueError):
        return None

    package_templates = profile.parent.parent
    return package_templates if package_templates.is_dir() else None


class PredictorArms(tuple[str, ...]):
    """Tuple-compatible predictor arms with a visible provenance marker."""

    provenance: str

    def __new__(
        cls, arms: tuple[str, ...], provenance: str
    ) -> "PredictorArms":
        result = super().__new__(cls, arms)
        result.provenance = provenance
        return result


def _profile_cofold(path: Path) -> Optional[dict[str, Any]]:
    """Return the nearest declared cofold block from a profile inheritance chain."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    overlay = data.get("overlay")
    top_level = overlay.get("top_level") if isinstance(overlay, dict) else None
    cofold = top_level.get("cofold") if isinstance(top_level, dict) else None
    if isinstance(cofold, dict):
        return cofold
    direct = data.get("cofold")
    if isinstance(direct, dict):
        return direct
    base_profile = data.get("base_profile")
    if not isinstance(base_profile, str) or not base_profile:
        return None
    return _profile_cofold(path.parent / base_profile)


def known_predictor_arms() -> PredictorArms:
    """Return tuple-compatible arm IDs with profile or fallback provenance."""
    template_dir = _template_dir()
    if template_dir is None:
        return PredictorArms(
            KNOWN_PREDICTOR_ARMS_FALLBACK,
            "fallback:KNOWN_PREDICTOR_ARMS_FALLBACK",
        )

    path = template_dir / "profiles" / "small-run.template.json"
    try:
        cofold = _profile_cofold(path)
        if not isinstance(cofold, dict):
            raise ValueError("small-run profile has no cofold block")
        arms = tuple(
            str(p["id"])
            for p in cofold["predictors"]
            if isinstance(p, dict) and p.get("enabled", True) is True
        )
        if arms:
            return PredictorArms(arms, f"profile:{path}")
    except (OSError, KeyError, TypeError, ValueError):
        return PredictorArms(
            KNOWN_PREDICTOR_ARMS_FALLBACK,
            "fallback:KNOWN_PREDICTOR_ARMS_FALLBACK",
        )
    return PredictorArms(
        KNOWN_PREDICTOR_ARMS_FALLBACK,
        "fallback:KNOWN_PREDICTOR_ARMS_FALLBACK",
    )


def template_rescore_seed_count() -> Optional[int]:
    """Seed count from the deployed small-run profile, or None.

    Reads len(cofold.rescore_seeds) out of the same template file that
    known_predictor_arms() reads. Returns None when the file is absent or
    malformed, because a count that scales cost must come from the user or
    from a read file, never from a constant here.
    """
    template_dir = _template_dir()
    if template_dir is None:
        return None

    path = template_dir / "profiles" / "small-run.template.json"
    try:
        cofold = _profile_cofold(path)
        if not isinstance(cofold, dict):
            return None
        seeds = cofold["rescore_seeds"]
        if isinstance(seeds, list) and seeds:
            if all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
                return len(seeds)
    except (OSError, KeyError, TypeError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Typed intent objects.
#
# Rule: every field either has no default, because guessing costs money, or
# a default that means "not stated", which every consumer treats as a
# question for the user rather than a value to pick.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    """What the run binds against.

    name is required. The dossier carries the structure path, residue map,
    chains, entities, and site residues the campaign config needs. Plain
    language never supplies those, so dossier_path starts as None and
    stays None until the user points at a dossier file.
    """

    name: str
    dossier_path: Optional[str] = None


@dataclass(frozen=True)
class IntentCounts:
    """Four different numbers, four different names.

    The RunIntent design separates them because the executor represents
    them at different levels: per-generator backbone counts, a global
    sequences-per-backbone, screen widths derived at run time, and
    selection.final_count for delivered designs. Only delivered_designs
    has no default. Saying "N equals 10" sets that one field and nothing
    else. The other three stay None unless actually stated or resolved.
    """

    delivered_designs: int                      # rows handed to the scientist
    generated_backbones: Optional[int] = None   # sum over generator arms
    sequences_per_backbone: Optional[int] = None
    screened_candidates: Optional[int] = None


@dataclass(frozen=True)
class SpendCeiling:
    """A user stated maximum spend.

    currency defaults to USD because the validator accepts nothing else;
    lane.py requires provider.budget.currency and the profiles use USD.
    amount_usd has no default. Equality with the estimate passes the
    budget check; the validator rejects only estimated > maximum
    (lane.py budget block, NEXT8.md correction).
    """

    amount_usd: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        if self.currency != "USD":
            raise ValueError("the executor accepts USD only")
        if not isinstance(self.amount_usd, (int, float)) or isinstance(self.amount_usd, bool):
            raise ValueError("amount_usd must be a number")
        if self.amount_usd <= 0:
            raise ValueError("amount_usd must be greater than zero")


@dataclass(frozen=True)
class Deadline:
    """When the user needs results.

    Exactly one of duration_days and date_iso may be set. There is no
    campaign-wide deadline field in the config today, so a Deadline rides
    along in the card and the freeze but lowers into nothing.
    """

    duration_days: Optional[int] = None
    date_iso: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.duration_days is None) == (self.date_iso is None):
            raise ValueError("set exactly one of duration_days and date_iso")
        if self.duration_days is not None and self.duration_days < 1:
            raise ValueError("duration_days must be at least 1")


@dataclass(frozen=True)
class BinderLength:
    """Positive ordered binder residue bounds."""

    minimum_length: int
    maximum_length: int

    def __post_init__(self) -> None:
        for name in ("minimum_length", "maximum_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_length > self.maximum_length:
            raise ValueError("minimum_length must not exceed maximum_length")


@dataclass(frozen=True)
class RunIntent:
    """Everything a user actually asked for. Nothing they did not."""

    target: TargetSpec
    counts: IntentCounts
    # Empty tuple means the user named no predictors. Consumers ask; they
    # never pick a default arm, because the choice changes cost and science.
    predictor_arms: tuple[str, ...] = ()
    # None means not stated. The profile's rescore seed list governs when
    # unset, and the approval card discloses that fact.
    seeds_per_design: Optional[int] = None
    binder_length: Optional[BinderLength] = None
    # None means no ceiling was stated. Approval refuses without one,
    # because spending without a ceiling cannot be gated.
    spend_ceiling: Optional[SpendCeiling] = None
    deadline: Optional[Deadline] = None
    provider_id: Optional[str] = None
    optimization_rounds: Optional[int] = None
    objective_metric: Optional[str] = None
    objective_direction: Optional[str] = None
    # Identity and provenance, excluded from the freeze digest.
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parsed_from: Optional[str] = None

    def __post_init__(self) -> None:
        if self.optimization_rounds is not None and (
            not isinstance(self.optimization_rounds, int)
            or isinstance(self.optimization_rounds, bool)
            or self.optimization_rounds < 1
        ):
            raise ValueError("optimization_rounds must be a positive integer")
        if self.objective_direction not in {None, "maximize", "minimize"}:
            raise ValueError("objective_direction must be maximize or minimize")
        if (self.objective_metric is None) != (self.objective_direction is None):
            raise ValueError("objective_metric and objective_direction must be set together")
        if self.provider_id is not None and not self.provider_id.strip():
            raise ValueError("provider_id must be a non-empty string")


# ---------------------------------------------------------------------------
# Refusals. Refusing is a feature, so refusals are structured values.
# ---------------------------------------------------------------------------


class AmbiguousRequest(Exception):
    """The sentence underspecifies the request. Carry the questions to ask."""

    def __init__(self, questions: list[str], understood: Optional[dict[str, Any]] = None) -> None:
        super().__init__("; ".join(questions))
        self.questions = questions
        self.understood = understood or {}


class LoweringBlocked(Exception):
    """The intent cannot become a runnable config yet. Carry reasons."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


# ---------------------------------------------------------------------------
# Parser: plain language to RunIntent.
# ---------------------------------------------------------------------------

# Number words are ordinary English, not domain facts, so spelling them out
# here adds no invented knowledge.
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "fifty": 50,
}
_VAGUE_QUANTITIES = {
    "a few": "a few", "few": "a few", "some": "some", "several": "several",
    "a handful": "a handful", "a couple": "a couple", "a bunch": "a bunch",
    "lots": "lots", "loads": "loads", "plenty": "plenty",
}

# A count may be a digit or a word. `_NUMBER_WORDS` above already carries the words,
# and `_word_or_int` already resolves either form, but the patterns below accepted
# digits only, so "ten of them" was refused as an unstated count. "10 of them" was
# refused too, because no pattern covered "of them" in any form.
_N_TOKEN = r"\d+|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))

_N_PATTERNS = [
    re.compile(rf"\bn\s*(?:=|equals|is)\s*({_N_TOKEN})\b", re.IGNORECASE),
    re.compile(rf"\b({_N_TOKEN})\s*(?:designs|binders|candidates)\b", re.IGNORECASE),
    re.compile(rf"\b(?:return(?:ing)?|give me|deliver|delivering|produce)\s+({_N_TOKEN})\b", re.IGNORECASE),
    re.compile(rf"\b({_N_TOKEN})\s+of\s+(?:them|these|those)\b", re.IGNORECASE),
]
_MONEY_TOKEN = r"(\d+(?:\.\d+)?)\s*([kK])?(?![A-Za-z])"
_MONEY_PATTERN = re.compile(
    r"(?:under|less than|below|up to|at most|maximum of|max\.?(?: of)?|no more than|not more than|<=|≤)"
    rf"\s*\$?\s*{_MONEY_TOKEN}\s*(?:usd|dollars?|bucks)?",
    re.IGNORECASE,
)
_BUDGET_OF_PATTERN = re.compile(
    rf"\bbudget\s+(?:of|is|:)?\s*\$?\s*{_MONEY_TOKEN}\s*(?:usd|dollars?)?\b",
    re.IGNORECASE,
)
_BARE_MONEY_PATTERN = re.compile(
    r"\$\s*\d+(?:\.\d+)?\s*[kK]?(?![A-Za-z])"
    r"|\b\d+(?:\.\d+)?\s*[kK]?\s*(?:usd|dollars?)\b",
    re.IGNORECASE,
)
_FOREIGN_CURRENCY = re.compile(r"\b(euros?|eur|pounds?|gbp|sterling|yen|jpy|cny|yuan)\b", re.IGNORECASE)
_SEEDS_PATTERN = re.compile(
    r"\b(\d+|[a-z]+)\s+seeds?\b(?:\s*(?:each|per\s+design|per\s+backbone))?", re.IGNORECASE
)
_LENGTH_PATTERN = re.compile(
    r"\b(\d+)\s*(?:-|to|through)\s*(\d+)\s*(?:residues?|aa|amino acids?)\b", re.IGNORECASE
)
_ROUNDS_PATTERN = re.compile(r"\b(\d+|[a-z]+)\s+(?:optimization\s+)?rounds?\b", re.IGNORECASE)
_OBJECTIVE_PATTERN = re.compile(
    r"\b(maximize|minimize)\s+([a-z][a-z0-9_\- ]{1,40}?)(?=\s*(?:,|;|\.|\bon\b|\bwith\b|\bunder\b|$))",
    re.IGNORECASE,
)
_PROVIDER_PATTERN = re.compile(r"\b(?:on|using|via)\s+(runpod|modal|lambda(?: cloud)?|fal)\b", re.IGNORECASE)
_DAYS_PATTERN = re.compile(r"\bwithin\s+(\d+|[a-z]+)\s+days?\b|\bin\s+(\d+|[a-z]+)\s+days?\b", re.IGNORECASE)
_WEEKS_PATTERN = re.compile(r"\bwithin\s+(\d+|[a-z]+)\s+weeks?\b", re.IGNORECASE)
_A_WEEK_PATTERN = re.compile(r"\bwithin\s+a\s+week\b|\bwithin\s+a\s+fortnight\b", re.IGNORECASE)
_DATE_PATTERN = re.compile(r"\bby\s+(\d{4}-\d{2}-\d{2})\b")
_TARGET_FOR = re.compile(
    r"\b(?:for|against|targeting?|bind|binding)\s+(?:the\s+)?"
    r"([a-z0-9][a-z0-9\-_]*(?:\s+[a-z0-9][a-z0-9\-_]*)?)",
    re.IGNORECASE,
)
_THIS_TARGET = re.compile(r"\bthis target\b|\bthat target\b", re.IGNORECASE)

# Presentation aliases for arm names people actually say. Each alias maps
# onto an ID read from the shipped profile; nothing new is invented here.
_ARM_ALIASES = {
    "esmfold2-fast": ("esmfold 2 fast", "esmfold fast", "fast esmfold", "fast"),
    "esmfold2": ("esmfold", "esmfold 2", "esm fold"),
    "protenix-v2": ("protenix", "protenix 2", "protenix v2"),
}

_OBJECTIVE_ALIASES = {
    "interfacepae": "interface_pae",
    "interface pae": "interface_pae",
    "interface-pae": "interface_pae",
}


def _word_or_int(token: str) -> Optional[int]:
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def _money_value(match: re.Match[str]) -> float:
    """Convert one matched amount, including an optional thousand suffix."""
    amount = float(match.group(1))
    return amount * 1000.0 if match.group(2) else amount


@dataclass
class ParseResult:
    """Outcome of parsing one utterance.

    ok is True only when an intent came back. Otherwise questions lists
    what to ask, and understood records what was safely captured so far.
    """

    ok: bool
    intent: Optional[RunIntent] = None
    questions: list[str] = field(default_factory=list)
    understood: dict[str, Any] = field(default_factory=dict)


def parse_request(
    text: str,
    *,
    context: Optional[dict[str, Any]] = None,
    known_arms: Optional[tuple[str, ...]] = None,
    policy: Optional[AccountPolicy] = None,
) -> ParseResult:
    """Parse one plain-language request into a RunIntent, or refuse.

    The parser writes a field only from words it actually matched. Vague
    quantities produce questions. Bare money amounts are refused because
    "200 dollars" could be a ceiling or a guess at cost, and the two have
    opposite failure modes.

    The parser has no provider tier, so it records the campaign ceiling without
    trying to compare it with an account policy. approve() applies the selected
    tier's cap once the plan names one. With no policy record, the explicit
    campaign ceiling is the only dollar cap.
    """
    context = context or {}
    arms = known_predictor_arms() if known_arms is None else known_arms
    questions: list[str] = []
    understood: dict[str, Any] = {}

    # Target name. "this target" resolves only against a bound dossier.
    target_name: Optional[str] = None
    dossier_path = context.get("dossier_path")
    target_name_hint = context.get("target_name")
    match = _THIS_TARGET.search(text)
    if match:
        if dossier_path:
            target_name = str(target_name_hint or dossier_path)
            understood["target"] = "this target (bound dossier)"
        else:
            questions.append(
                "Which target do you mean? Bind a target dossier first, or name the target."
            )
    else:
        match = _TARGET_FOR.search(text)
        if match:
            raw = match.group(1)
            # Stop the capture at common clause boundaries.
            raw = re.split(r"\s+(?:with|using|under|within|and|return|budget|n\b)", raw, flags=re.IGNORECASE)[0]
            target_name = raw.strip()
            understood["target"] = target_name
        else:
            questions.append("Which target should the binders bind? Name it or point at a dossier.")

    # Delivered designs. The four counts stay separate; only this one is
    # parsed, because "N" in the product goal means what comes back.
    n_delivered: Optional[int] = None
    for pattern in _N_PATTERNS:
        match = pattern.search(text)
        if match:
            # `_word_or_int` returns None for a token it cannot resolve. Treat that as
            # no match rather than a zero, so the count stays unstated and gets asked for.
            parsed_n = _word_or_int(match.group(1))
            if parsed_n is None:
                continue
            n_delivered = parsed_n
            understood["delivered_designs"] = n_delivered
            break
    if n_delivered is None:
        found_vague = next((v for k, v in _VAGUE_QUANTITIES.items() if re.search(re.escape(k), text, re.IGNORECASE)), None)
        if found_vague:
            understood["delivered_designs"] = f'unstated ({found_vague})'
        questions.append(
            "How many delivered designs do you want back? Give an exact number, for example 10."
        )
    elif n_delivered < 1:
        questions.append("The design count must be at least 1. How many do you want?")

    # Spend ceiling. Comparators only. A bare amount asks a question.
    ceiling_amount: Optional[float] = None
    foreign = _FOREIGN_CURRENCY.search(text)
    if foreign:
        questions.append(
            f'The executor accepts USD only (provider.budget.currency). What is your ceiling in USD?'
        )
    money_matches = sorted(
        [*_MONEY_PATTERN.finditer(text), *_BUDGET_OF_PATTERN.finditer(text)],
        key=lambda found: found.start(),
    )
    if len(money_matches) > 1:
        # A per-wave figure and a total figure have different meanings. Refuse both
        # rather than choosing the first match and silently lowering the wrong ceiling.
        candidates = [_money_value(found) for found in money_matches]
        understood["spend_ceiling_candidates_usd"] = candidates
        questions.append(
            "I found multiple spend ceilings ("
            + ", ".join(f"{value:.2f} USD" for value in candidates)
            + "). State one total USD ceiling for this run."
        )
    elif money_matches:
        ceiling_amount = _money_value(money_matches[0])
        understood["spend_ceiling_usd"] = ceiling_amount
    elif _BARE_MONEY_PATTERN.search(text):
        questions.append(
            'I saw a money amount without "under" or "at most". '
            "Is that amount your maximum spend? Say \"under 200 dollars\" to set a ceiling."
        )
    elif not foreign:
        questions.append("What is your maximum spend? For example: under 200 dollars.")
    # Seeds per design. Baseline-specific replication is checked during
    # lowering, once the selected profile is known.
    seeds: Optional[int] = None
    match = _SEEDS_PATTERN.search(text)
    if match:
        value = _word_or_int(match.group(1))
        if value is None:
            questions.append('How many seeds per design? "a few seeds" is ambiguous; give a number.')
        else:
            seeds = value
            understood["seeds_per_design"] = seeds

    # Predictor arms. Matched by ID or alias. Longer names win: "esmfold2"
    # sits inside "esmfold2-fast", and a plain word-boundary search would
    # otherwise claim both arms from one mention. Unmatched tool-like words
    # after "use" or "with" become a question rather than a guess.
    lowered_text = text.lower()
    matched_spans: list[tuple[int, int]] = []
    matched_arms: set[str] = set()

    def _span_inside(spans: list[tuple[int, int]], span: tuple[int, int]) -> bool:
        return any(start <= span[0] and span[1] <= end for start, end in spans)

    for arm_id in sorted(arms, key=len, reverse=True):
        hit = re.search(rf"\b{re.escape(arm_id)}\b", lowered_text)
        if hit is None:
            for alias in _ARM_ALIASES.get(arm_id, ()):  # presentation aliases only
                hit = re.search(rf"(?<![a-z0-9-]){re.escape(alias)}(?![a-z0-9-])", lowered_text)
                if hit:
                    break
        if hit is None:
            continue
        span = (hit.start(), hit.end())
        if _span_inside(matched_spans, span):
            continue
        matched_arms.add(arm_id)
        matched_spans.append(span)
    stated_arms = [arm_id for arm_id in arms if arm_id in matched_arms]
    used_with = re.search(r"\b(?:use|using|with)\s+([a-z0-9\- ]{3,40}?)(?:\.|,|;|$)", text, re.IGNORECASE)
    if used_with:
        fragment = used_with.group(1).strip().lower()
        known_hit = any(a in fragment for a in arms) or any(
            alias in fragment for a in arms for alias in _ARM_ALIASES.get(a, ())
        )
        looks_like_tool = re.search(r"\b(mpnn|rfdiffusion|genie|boltz|chai|openfold|alphafold)\b", fragment)
        if not known_hit and looks_like_tool:
            questions.append(
                '"use ' + fragment + '" does not name a predictor arm I know. '
                "Known predictor arms: " + ", ".join(arms) + "."
            )
    if stated_arms:
        understood["predictor_arms"] = stated_arms
    elif isinstance(context.get("predictor_arms"), (list, tuple)):
        stated_arms = [
            str(value)
            for value in context["predictor_arms"]
            if isinstance(value, str) and value
        ]
        if stated_arms:
            understood["predictor_arms"] = {
                "values": stated_arms,
                "source": "resolved campaign",
            }

    # Binder length bounds.
    binder_length: Optional[BinderLength] = None
    match = _LENGTH_PATTERN.search(text)
    if match:
        lo, hi = int(match.group(1)), int(match.group(2))
        try:
            binder_length = BinderLength(lo, hi)
            understood["binder_length"] = f"{lo}-{hi}"
        except ValueError as exc:
            questions.append(f"The length range {lo}-{hi} is invalid: {exc}")
    elif isinstance(context.get("binder_length"), (list, tuple)) and len(context["binder_length"]) == 2:
        try:
            binder_length = BinderLength(*context["binder_length"])
            understood["binder_length"] = {
                "value": f"{binder_length.minimum_length}-{binder_length.maximum_length}",
                "source": "resolved campaign",
            }
        except (TypeError, ValueError):
            binder_length = None

    optimization_rounds: Optional[int] = None
    match = _ROUNDS_PATTERN.search(text)
    if match:
        optimization_rounds = _word_or_int(match.group(1))
        if optimization_rounds is None or optimization_rounds < 1:
            questions.append("How many optimization rounds? Give a positive integer.")
            optimization_rounds = None
        else:
            understood["optimization_rounds"] = optimization_rounds

    objective_metric: Optional[str] = None
    objective_direction: Optional[str] = None
    match = _OBJECTIVE_PATTERN.search(text)
    if match:
        objective_direction = match.group(1).lower()
        raw_metric = " ".join(match.group(2).strip().lower().split())
        objective_metric = _OBJECTIVE_ALIASES.get(raw_metric, raw_metric.replace("-", "_"))
        understood["objective"] = {
            "metric": objective_metric,
            "direction": objective_direction,
        }

    provider_id: Optional[str] = None
    match = _PROVIDER_PATTERN.search(text)
    if match:
        provider_id = match.group(1).lower().replace(" cloud", "")
        understood["provider_id"] = provider_id

    # Deadline, as relative days or a fixed date.
    deadline: Optional[Deadline] = None
    match = _A_WEEK_PATTERN.search(text)
    if match:
        days = 14 if "fortnight" in match.group(0) else 7
        deadline = Deadline(duration_days=days)
        understood["deadline"] = f"{days} days"
    if deadline is None:
        match = _WEEKS_PATTERN.search(text)
        if match:
            weeks = _word_or_int(match.group(1))
            if weeks is None:
                questions.append('How many weeks? Give a number, for example "within 2 weeks".')
            else:
                deadline = Deadline(duration_days=int(weeks) * 7)
                understood["deadline"] = f"{weeks * 7} days"
    if deadline is None:
        match = _DAYS_PATTERN.search(text)
        if match:
            days = _word_or_int(match.group(1) or match.group(2))
            if days is None:
                questions.append('How many days? Give a number, for example "within two days".')
            else:
                deadline = Deadline(duration_days=days)
                understood["deadline"] = f"{days} days"
    if deadline is None:
        match = _DATE_PATTERN.search(text)
        if match:
            deadline = Deadline(date_iso=match.group(1))
            understood["deadline"] = match.group(1)

    # Build the intent only when nothing is missing. Any question means
    # refusal: the parser never guesses a value it did not read.
    if questions:
        return ParseResult(ok=False, questions=questions, understood=understood)

    intent = RunIntent(
        target=TargetSpec(name=target_name or "unknown", dossier_path=dossier_path),
        counts=IntentCounts(delivered_designs=n_delivered),
        predictor_arms=tuple(stated_arms),
        seeds_per_design=seeds,
        binder_length=binder_length,
        spend_ceiling=SpendCeiling(ceiling_amount),
        deadline=deadline,
        provider_id=provider_id,
        optimization_rounds=optimization_rounds,
        objective_metric=objective_metric,
        objective_direction=objective_direction,
        parsed_from=text,
    )
    return ParseResult(ok=True, intent=intent, understood=understood)


# ---------------------------------------------------------------------------
# Lowering: RunIntent to the campaign config the lane accepts.
#
# The overlay written here contains only fields the intent owns. Everything
# else is reported as cannot_fill with a reason, because inventing controls,
# thresholds, adapter revisions, or stage graphs would fabricate science.
# ---------------------------------------------------------------------------

# Config locations this module can fill, with the executor rule that governs
# each. Kept as data so the report and the freeze agree by construction.
_FILL_MAP = {
    "counts.delivered_designs": "selection.final_count",
    "counts.sequences_per_backbone": "sequence_design.sequences_per_backbone",
    "counts.generated_backbones": "generation.generators[*].backbone_count (distributed)",
    "predictor_arms": "cofold.predictors[*].enabled",
    "binder_length": "binder.minimum_length, binder.maximum_length",
    "spend_ceiling": "provider.budget.maximum_spend_usd",
}


@dataclass
class LoweringReport:
    """What lowering filled, what it refused to fill, and why."""

    overlay: dict[str, Any]
    filled: dict[str, str]
    cannot_fill: dict[str, str]

    @property
    def blocked(self) -> list[str]:
        return sorted(self.cannot_fill)


def write_intent_overlay(path: Path, report: LoweringReport) -> None:
    """Write the partial config that must be applied before composition."""
    path.write_text(
        json.dumps(report.overlay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def lower_intent(
    intent: RunIntent,
    *,
    profile: Optional[dict[str, Any]] = None,
    campaign_overrides: Optional[Mapping[str, Any]] = None,
) -> LoweringReport:
    """Lower a validated intent into a partial campaign config overlay.

    ``profile`` or ``campaign_overrides`` may carry choices already resolved by
    Claude Science, including external predictor records and provider details.
    The lowerer preserves those records and applies only values the scientist
    stated. It does not rebuild the campaign schema.
    """
    filled: dict[str, str] = {}
    cannot_fill: dict[str, str] = {}
    overlay: dict[str, Any] = copy.deepcopy(dict(campaign_overrides or {}))
    configured: Mapping[str, Any] = campaign_overrides or profile or {}
    profile_block = configured.get("profile")
    baseline = (
        isinstance(profile_block, Mapping)
        and profile_block.get("baseline_fidelity") is True
    )

    # N delivered designs lands in selection.final_count.
    overlay.setdefault("selection", {})["final_count"] = intent.counts.delivered_designs
    filled["counts.delivered_designs"] = "selection.final_count"

    # Sequences per backbone only when the user said it. The shipped
    # profile uses 2; leaving it alone when unstated keeps the profile
    # authoritative instead of restating its value here.
    if intent.counts.sequences_per_backbone is not None:
        overlay.setdefault("sequence_design", {})[
            "sequences_per_backbone"
        ] = intent.counts.sequences_per_backbone
        filled["counts.sequences_per_backbone"] = "sequence_design.sequences_per_backbone"
    else:
        cannot_fill["sequence_design.sequences_per_backbone"] = (
            "not stated by the user; the execution profile owns this value"
        )

    # Aggregate backbone count distributes across generator arms only when
    # both the aggregate and the profile's generator list are known.
    if intent.counts.generated_backbones is not None:
        generation = configured.get("generation")
        generator_rows = generation.get("generators") if isinstance(generation, Mapping) else None
        generators = [
            g
            for g in (generator_rows if isinstance(generator_rows, list) else [])
            if isinstance(g, Mapping) and g.get("enabled", True)
        ]
        if generators:
            total = intent.counts.generated_backbones
            even, remainder = divmod(total, len(generators))
            distribution = {g["id"]: even for g in generators}
            distribution[generators[0]["id"]] += remainder
            overlay.setdefault("generation", {})["backbone_distribution"] = distribution
            filled["counts.generated_backbones"] = (
                "generation.generators[*].backbone_count (distribution: "
                + json.dumps(distribution, sort_keys=True) + ")"
            )
        else:
            cannot_fill["generation.generators[*].backbone_count"] = (
                "an aggregate count needs the profile's generator list to distribute across arms"
            )
    else:
        cannot_fill["generation.generators[*].backbone_count"] = (
            "not stated by the user; per-arm backbone counts come from the profile"
        )

    # Predictor arms are resolved against the selected campaign or profile,
    # rather than a built-in allowlist. This permits platform-native and custom
    # adapters after Claude Science has described their contract.
    cofold = configured.get("cofold")
    if intent.predictor_arms:
        predictor_rows = cofold.get("predictors") if isinstance(cofold, Mapping) else None
        if isinstance(predictor_rows, list):
            known_ids = {
                str(p["id"]): p
                for p in predictor_rows
                if isinstance(p, Mapping) and isinstance(p.get("id"), str)
            }
            unknown = [arm for arm in intent.predictor_arms if arm not in known_ids]
            if unknown:
                cannot_fill["cofold.predictors[*]"] = (
                    "selected campaign has no predictor record for: " + ", ".join(unknown)
                    + ". Add the resolved native, hosted, or adapter route before composition."
                )
            if not unknown:
                switches = {}
                for pid, record in known_ids.items():
                    want = pid in intent.predictor_arms
                    if record.get("enabled", True) is not want:
                        switches[pid] = want
                if switches:
                    overlay.setdefault("cofold", {})["enable_predictors"] = switches
                filled["predictor_arms"] = "cofold.predictors[*].enabled (" + ", ".join(
                    sorted(intent.predictor_arms)
                ) + ")"
        else:
            cannot_fill["cofold.predictors[*].enabled"] = (
                "the selected campaign or profile must supply predictor records, including "
                "the adapter or native-route handoff for each requested arm"
            )
    else:
        cannot_fill["cofold.predictors[*].enabled"] = (
            "user named no predictor arms; asking beats choosing, because arms change cost and ranking"
        )

    # Custom campaigns can use any nonempty distinct rescore set. Exact baseline
    # fidelity retains the published five-seed minimum.
    if intent.seeds_per_design is not None:
        minimum = MIN_BASELINE_RESCORE_SEEDS if baseline else MIN_CUSTOM_RESCORE_SEEDS
        if intent.seeds_per_design < minimum:
            raise LoweringBlocked([
                f"seeds_per_design={intent.seeds_per_design} conflicts with the executor, "
                f"which requires at least {minimum} distinct rescore seeds for "
                f"{'published baseline fidelity' if baseline else 'this custom campaign'}."
            ])
        configured_seeds = cofold.get("rescore_seeds") if isinstance(cofold, Mapping) else None
        if (
            isinstance(configured_seeds, list)
            and len(configured_seeds) == intent.seeds_per_design
            and len(set(configured_seeds)) == len(configured_seeds)
            and all(isinstance(seed, int) and not isinstance(seed, bool) for seed in configured_seeds)
        ):
            seeds = list(configured_seeds)
        else:
            seeds = list(range(intent.seeds_per_design))
        overlay.setdefault("cofold", {})["rescore_seeds"] = seeds
        filled["seeds_per_design"] = "cofold.rescore_seeds"
        scoring_overlay = overlay.setdefault("scoring", {})
        configured_scoring = configured.get("scoring")
        configured_minimum = (
            configured_scoring.get("minimum_seed_observations")
            if isinstance(configured_scoring, Mapping)
            else None
        )
        if (
            isinstance(configured_minimum, int)
            and not isinstance(configured_minimum, bool)
            and configured_minimum > intent.seeds_per_design
        ):
            scoring_overlay["minimum_seed_observations"] = intent.seeds_per_design
            filled["minimum_seed_observations"] = (
                "scoring.minimum_seed_observations capped at the requested rescore count"
            )
    else:
        cannot_fill["cofold.rescore_seeds"] = (
            "not stated by the user; the profile's seed list governs and the card discloses it"
        )

    # Binder lengths.
    if intent.binder_length is not None:
        overlay.setdefault("binder", {})["minimum_length"] = intent.binder_length.minimum_length
        overlay["binder"]["maximum_length"] = intent.binder_length.maximum_length
        filled["binder_length"] = "binder.minimum_length, binder.maximum_length"
    else:
        cannot_fill["binder.minimum_length"] = (
            "not stated by the user; use the selected tool's supported range"
        )
        cannot_fill["binder.maximum_length"] = "not stated by the user"

    if intent.provider_id is not None:
        overlay.setdefault("provider", {})["provider_id"] = intent.provider_id
        filled["provider_id"] = "provider.provider_id"

    if intent.optimization_rounds is not None:
        optimization = overlay.setdefault("optimization", {})
        optimization["enabled"] = True
        optimization["rounds"] = intent.optimization_rounds
        filled["optimization_rounds"] = "optimization.enabled, optimization.rounds"

    if intent.objective_metric is not None:
        if baseline and (
            intent.objective_metric != "ipsae_min"
            or intent.objective_direction != "maximize"
        ):
            raise LoweringBlocked([
                "a custom optimization objective conflicts with published baseline fidelity; "
                "select a custom campaign or keep the published objective"
            ])
        scoring = overlay.setdefault("scoring", {})
        scoring["primary_metric"] = intent.objective_metric
        scoring.setdefault("metric_directions", {})[
            intent.objective_metric
        ] = intent.objective_direction
        if not baseline:
            scoring["ranking_mode"] = "custom-weighted-zscore"
        filled["objective"] = (
            "scoring.primary_metric, scoring.metric_directions, scoring.ranking_mode"
        )
        configured_scoring = configured.get("scoring")
        configured_weights = (
            configured_scoring.get("rank_weights")
            if isinstance(configured_scoring, Mapping)
            else None
        )
        pose_metric = (
            configured_scoring.get("pose_metric", "sc_dockq")
            if isinstance(configured_scoring, Mapping)
            else "sc_dockq"
        )
        required_weight_keys = {
            f"{metric}_z" for metric in {intent.objective_metric, str(pose_metric)}
        }
        if not isinstance(configured_weights, Mapping) or not required_weight_keys.issubset(
            configured_weights
        ):
            cannot_fill["scoring.rank_weights"] = (
                "the selected campaign must state positive weights for the chosen objective "
                f"and pose metric ({', '.join(sorted(required_weight_keys))})"
            )

    # Spend ceiling. The graph estimator writes its own approval record after
    # materialization, so a profile-level estimate is optional provenance and
    # is not a missing user decision.
    if intent.spend_ceiling is not None:
        overlay.setdefault("provider", {}).setdefault("budget", {})[
            "maximum_spend_usd"
        ] = intent.spend_ceiling.amount_usd
        overlay["provider"]["budget"]["currency"] = intent.spend_ceiling.currency
        filled["spend_ceiling"] = "provider.budget.maximum_spend_usd"
        cannot_fill["provider.budget.pricing_source"] = (
            "must name the provider price record read at run time; the operator attests its source"
        )
        cannot_fill["provider.budget.pricing_checked_at"] = (
            "timestamped when a real price record is read, never now-ish made up"
        )
    else:
        cannot_fill["provider.budget.maximum_spend_usd"] = (
            "no ceiling stated; approval refuses without one"
        )

    # Deadline. The config has no campaign-wide duration field, only
    # per-stage timeout_minutes and lifecycle knobs, so this maps nowhere.
    if intent.deadline is not None:
        cannot_fill["schedule/deadline"] = (
            "the campaign config has no campaign-wide deadline field; the deadline "
            "stays on the card and in the freeze"
        )

    # Licence. claude_binder/gate.py requires declared_use and refuses an
    # undeclared campaign. No plain-language request states it, and a guess here
    # would guess a licence, so the question is asked at this step rather than
    # met as a refusal at composition.
    if configured.get("declared_use") is None:
        cannot_fill["declared_use"] = (
            'not stated by the user; set it to "commercial" or "non-commercial" before '
            'composing. "non-commercial" checks only that every selected tool has a catalog '
            'entry. "commercial" also requires each selected tool to carry cited '
            "commercial-use terms for its code and its weights, and refuses the campaign "
            "when one does not"
        )

    # Target science. Plain language gives a name at most. Structure path,
    # residue map, chains, entities, and site residues come from a dossier.
    configured_targets = configured.get("targets")
    if not isinstance(configured_targets, list) or not configured_targets:
        cannot_fill["targets[0]"] = (
            "needs a target dossier (structure_path, residue_map_path, chains, entities, site); "
            f"intent.target.dossier_path is {'set' if intent.target.dossier_path else 'not set'}"
        )
    cannot_fill["campaign composition"] = (
        "compose validates the selected controls, filters, adapters, stages, and provider "
        "lifecycle. Values already supplied in campaign_overrides are preserved."
    )

    return LoweringReport(overlay=overlay, filled=filled, cannot_fill=cannot_fill)


# ---------------------------------------------------------------------------
# Constraint Freeze: a stable digest over intent plus resolved settings.
#
# Canonical form follows the packet design: NFC strings, integer-valued
# floats collapsed, object keys sorted, ordered arrays kept in order, and
# set-valued arrays sorted. Timestamps and identity fields are excluded.
# The byte serialization mirrors sha256_json in lane.py (sorted keys,
# compact separators, UTF-8).
# ---------------------------------------------------------------------------

FREEZE_SCHEMA_TAG = "run-intent-constraint-freeze-v1"

# Array fields whose meaning is a set. Order there is cosmetic, so they are
# sorted before hashing. Everything else keeps user order.
_SORTED_LIST_FIELDS = {
    "requested.predictor_arms",
    "resolved.enabled_predictor_arms",
}


def _canonicalize(value: Any, path: str = "") -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        items = [_canonicalize(item, f"{path}[{i}]") for i, item in enumerate(value)]
        if path in _SORTED_LIST_FIELDS:
            return sorted(items, key=lambda v: json.dumps(v, sort_keys=True))
        return items
    if isinstance(value, dict):
        return {k: _canonicalize(v, f"{path}.{k}") for k, v in sorted(value.items())}
    raise TypeError(f"freeze projection cannot hold {type(value).__name__} at {path}")


def _file_sha256(path_str: str) -> Optional[str]:
    """Lowercase hex SHA-256 of a file's bytes, mirroring sha256_file in lane.py."""
    path = Path(path_str)
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


@dataclass(frozen=True)
class ConstraintFreeze:
    digest: str
    projection: dict[str, Any]


def freeze_constraints(
    intent: RunIntent,
    report: LoweringReport,
    *,
    resolved_seeds: Optional[list[int]] = None,
    enabled_arms: Optional[list[str]] = None,
    dossier_hashes: Optional[dict[str, str]] = None,
) -> ConstraintFreeze:
    """Hash the intent together with the settings a run would actually use.

    Cosmetic changes (key order, unicode form, 10 vs 10.0) leave the digest
    unchanged. Scientific changes (counts, arms, lengths, ceiling, seeds,
    target bytes) change it.
    """
    projection = {
        "schema": FREEZE_SCHEMA_TAG,
        "requested": {
            "target_name": intent.target.name,
            "target_dossier_sha256": (
                dossier_hashes
                or (_dossier_hash(intent.target.dossier_path) if intent.target.dossier_path else None)
            ),
            "delivered_designs": intent.counts.delivered_designs,
            "generated_backbones": intent.counts.generated_backbones,
            "sequences_per_backbone": intent.counts.sequences_per_backbone,
            "screened_candidates": intent.counts.screened_candidates,
            "predictor_arms_requested": list(intent.predictor_arms),
            "seeds_per_design_stated": intent.seeds_per_design,
            "binder_minimum_length": intent.binder_length.minimum_length if intent.binder_length else None,
            "binder_maximum_length": intent.binder_length.maximum_length if intent.binder_length else None,
            "maximum_spend_usd": intent.spend_ceiling.amount_usd if intent.spend_ceiling else None,
            "currency": intent.spend_ceiling.currency if intent.spend_ceiling else None,
            "deadline_duration_days": intent.deadline.duration_days if intent.deadline else None,
            "deadline_date_iso": intent.deadline.date_iso if intent.deadline else None,
            "provider_id": intent.provider_id,
            "optimization_rounds": intent.optimization_rounds,
            "objective_metric": intent.objective_metric,
            "objective_direction": intent.objective_direction,
        },
        "resolved": {
            "final_count": report.overlay.get("selection", {}).get("final_count"),
            "sequences_per_backbone": report.overlay.get("sequence_design", {}).get(
                "sequences_per_backbone"
            ),
            "backbone_distribution": report.overlay.get("generation", {}).get(
                "backbone_distribution"
            ),
            "enabled_predictor_arms": sorted(enabled_arms or list(intent.predictor_arms)),
            "rescore_seed_list_source": (
                "user-count-only; specific seeds owned by the execution profile"
                if intent.seeds_per_design is not None
                else "execution-profile default"
            ),
            "resolved_rescore_seeds": sorted(resolved_seeds) if resolved_seeds else None,
            "maximum_spend_usd_written": report.overlay.get("provider", {}).get(
                "budget", {}
            ).get("maximum_spend_usd"),
            "provider_id": report.overlay.get("provider", {}).get("provider_id"),
            "optimization_rounds": report.overlay.get("optimization", {}).get("rounds"),
            "optimization_enabled": report.overlay.get("optimization", {}).get("enabled"),
            "primary_metric": report.overlay.get("scoring", {}).get("primary_metric"),
            "metric_directions": report.overlay.get("scoring", {}).get("metric_directions"),
        },
        # Excluded on purpose: intent_id, timestamps, prose, local paths.
    }
    payload = json.dumps(
        _canonicalize(projection), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return ConstraintFreeze(digest=hashlib.sha256(payload).hexdigest(), projection=projection)


def _dossier_hash(path_str: str) -> Optional[dict[str, str]]:
    single = _file_sha256(path_str)
    if single is not None:
        return {"dossier_file": single}
    return None


def verify_freeze(expected_digest: str, intent: RunIntent, report: LoweringReport, **kwargs: Any) -> bool:
    """Recompute the freeze and compare. Used before a re-run spends again."""
    return freeze_constraints(intent, report, **kwargs).digest == expected_digest


# ---------------------------------------------------------------------------
# Cost estimate. GPU-second arithmetic on measured units. Dollars need a
# runtime rate parameter and stay UNKNOWN without one.
# ---------------------------------------------------------------------------


@dataclass
class Estimate:
    """An estimate, clearly separated from any measurement.

    GPU and server-time fields carry bounds because the fal receipts vary by
    seed. Dollars remain None until a caller passes a runtime rate. Modal
    timings remain None until a Modal measurement supplies them.
    """

    designs: int
    seeds_per_design: int
    folds: int
    shard_size: Optional[int]
    containers: Optional[int]
    gpu_seconds_min: Optional[float]
    gpu_seconds_max: Optional[float]
    server_seconds_min: Optional[float]
    server_seconds_max: Optional[float]
    seconds_kind: str
    tier: str
    usd_estimate_min: Optional[float] = None
    usd_estimate_max: Optional[float] = None
    rate_status: str = "UNKNOWN"
    rate_source: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    stage_count: int = 0
    paid_stage_count: int = 0
    covered_stage_ids: list[str] = field(default_factory=list)
    upper_bound_stage_ids: list[str] = field(default_factory=list)
    unknown_stage_ids: list[str] = field(default_factory=list)
    stage_estimates: list[dict[str, Any]] = field(default_factory=list)
    workload_units: int = 0
    provider_records_required: bool = False

    @property
    def gpu_seconds(self) -> Optional[float]:
        """Return a scalar only when the measured bounds are identical."""
        if self.gpu_seconds_min == self.gpu_seconds_max:
            return self.gpu_seconds_min
        return None

    @property
    def usd_estimate(self) -> Optional[float]:
        """Return a scalar only when the dollar bounds are identical."""
        if self.usd_estimate_min == self.usd_estimate_max:
            return self.usd_estimate_min
        return None


@dataclass(frozen=True)
class RateRecord:
    """Operator-attested per-machine planning prices for one approval attempt.

    A provider rate display or published price may supply this conservative
    admission-control input when its source and read time are recorded.  It is
    not a settled bill and must not be described as one.
    """

    rates_usd_per_second: dict[str, float]
    source: str
    read_at: str

    @property
    def source_label(self) -> str:
        return f"{self.source}; read_at={self.read_at}"


def load_rate_record(path: Path) -> RateRecord:
    """Load a source- and timestamp-bearing planning-rate record.

    The record is explicit input. Callers that omit it continue with an
    UNKNOWN price instead of inheriting a repository default.  ``read_at`` is
    provenance in this format; the current loader does not authenticate the
    source or enforce freshness, so the operator remains responsible for
    attesting both before approval.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read rate record {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("rate record must be a JSON object")
    source = value.get("source")
    read_at = value.get("read_at")
    entries = value.get("rates")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("rate record source must be a non-empty string")
    if not isinstance(read_at, str) or not read_at.strip():
        raise ValueError("rate record read_at must be a non-empty timestamp string")
    if not isinstance(entries, list) or not entries:
        raise ValueError("rate record rates must be a non-empty list")
    rates: dict[str, float] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"rate record rates[{index}] must be an object")
        machine = entry.get("machine")
        rate = _finite_bound(entry.get("rate_usd_per_second"))
        if not isinstance(machine, str) or not machine:
            raise ValueError(f"rate record rates[{index}].machine must be a non-empty string")
        if rate is None or rate == 0:
            raise ValueError(f"rate record rates[{index}].rate_usd_per_second must be positive")
        if machine in rates:
            raise ValueError(f"rate record repeats machine: {machine}")
        rates[machine] = rate
    return RateRecord(rates_usd_per_second=rates, source=source.strip(), read_at=read_at.strip())


def _finite_bound(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _bounds(value: Any) -> tuple[Optional[float], Optional[float]]:
    """Read a lower and upper bound from a timing record."""
    if isinstance(value, dict):
        lower = next(
            (_finite_bound(value.get(key)) for key in ("minimum", "min", "lower", "seconds_min")
             if _finite_bound(value.get(key)) is not None),
            None,
        )
        upper = next(
            (_finite_bound(value.get(key)) for key in ("maximum", "max", "upper", "seconds_max")
             if _finite_bound(value.get(key)) is not None),
            None,
        )
        if lower is None and upper is not None:
            lower = upper
        if upper is None and lower is not None:
            upper = lower
        return lower, upper
    scalar = _finite_bound(value)
    return scalar, scalar


def _stage_list(plan: dict[str, Any]) -> list[dict[str, Any]]:
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ValueError("materialized plan must contain a stages list")
    return [stage for stage in stages if isinstance(stage, dict)]


def plan_final_count(plan: dict[str, Any]) -> int:
    """Read the delivered-design count recorded in a materialized plan."""
    selection = plan.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("materialized plan has no selection object")
    final_count = selection.get("final_count")
    if isinstance(final_count, bool) or not isinstance(final_count, int) or final_count < 1:
        raise ValueError("materialized plan selection.final_count must be a positive integer")
    return final_count


def validate_approval_design_counts(
    intent: RunIntent,
    plan: dict[str, Any],
    estimate: Estimate,
) -> None:
    """Refuse an approval when its recorded request differs from its plan or estimate."""
    plan_count = plan_final_count(plan)
    recorded_count = intent.counts.delivered_designs
    blocks: list[str] = []
    if recorded_count != plan_count:
        blocks.append(
            f"requested.delivered_designs={recorded_count} differs from "
            f"plan selection.final_count={plan_count}"
        )
    if estimate.designs != recorded_count:
        blocks.append(
            f"estimate.designs={estimate.designs} differs from "
            f"requested.delivered_designs={recorded_count}"
        )
    if blocks:
        raise LoweringBlocked(blocks)


def _lookup_stage_map(plan: dict[str, Any], names: tuple[str, ...], key: str) -> Any:
    for name in names:
        mapping = plan.get(name)
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return None


def _stage_adapter(plan: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    adapter_id = stage.get("adapter_id")
    for adapter in plan.get("adapters", []):
        if isinstance(adapter, dict) and adapter.get("adapter_id") == adapter_id:
            return adapter
    return {}


def _has_selected_external_provider_route(
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> bool:
    """Return whether materialized stage or adapter fields select an external route."""
    for item in (stage, adapter):
        for key in ("provider_id", "execution_provider"):
            provider_id = _canonical_provider_id(item.get(key))
            if provider_id is not None and provider_id != "local":
                return True
        provider = item.get("provider")
        if isinstance(provider, Mapping):
            provider_id = _canonical_provider_id(provider.get("provider_id"))
            if provider_id is not None and provider_id != "local":
                return True

    environment = adapter.get("environment")
    if isinstance(environment, Mapping):
        if environment.get("CLAUDE_BINDER_EXECUTION_ROUTE") == "modal-platform":
            return True
        if any(
            isinstance(environment.get(key), str) and bool(environment[key].strip())
            for key in (
                "RFDIFFUSION3_FAL_URL",
                "PROTEINMPNN_FAL_URL",
                "ESMFOLD2_FAST_FAL_URL",
                "ALPHAFOLD_MULTIMER_V3_FAL_URL",
            )
        ):
            return True

    argv = adapter.get("command_argv_template")
    command = argv if isinstance(argv, list) else []
    protocol_present, protocol = _provider_argv_option(command, "--runner-protocol")
    if protocol_present and protocol not in {"auto", "local"}:
        return True
    fal_url_present, _ = _provider_argv_option(command, "--fal-url")
    if fal_url_present:
        return True
    if "-m" in command:
        module_index = command.index("-m")
        if module_index + 1 < len(command):
            module_name = command[module_index + 1]
            if isinstance(module_name, str) and module_name.rsplit(".", 1)[-1].startswith("fal_"):
                return True
    return False


def _declared_gpu_count(adapter: Mapping[str, Any]) -> Optional[float]:
    resources = adapter.get("resources")
    if not isinstance(resources, Mapping):
        return None
    gpu = resources.get("gpu")
    if isinstance(gpu, (int, float)) and not isinstance(gpu, bool):
        return float(gpu)
    return None


def _stage_is_paid(stage: dict[str, Any], adapter: dict[str, Any]) -> bool:
    """Classify spend using only the documented ``provider_facing`` override.

    Provider selection, a positive GPU request, or any ``provider_facing=true``
    declaration wins over false declarations. Legacy ``paid``, ``billable``,
    ``cost.paid``, and ``free`` keys are intentionally not classification inputs.
    """
    if any(item.get("provider_facing") is True for item in (stage, adapter)):
        return True
    if _has_selected_external_provider_route(stage, adapter):
        return True
    if adapter.get("adapter_id") in LOCAL_NO_PROVIDER_ADAPTER_IDS:
        return False
    gpu = _declared_gpu_count(adapter)
    if gpu is not None and gpu > 0:
        return True
    if (
        adapter.get("provider_facing") is False
        and gpu == 0
        and adapter.get("role") in PROVIDER_FACING_ADAPTER_ROLES
    ):
        return False
    if adapter.get("role") in PROVIDER_FACING_ADAPTER_ROLES:
        return True
    if any(item.get("provider_facing") is False for item in (stage, adapter)):
        return False
    if gpu is not None:
        return gpu > 0
    # A stage without an explicit local declaration stays in the paid set. This
    # prevents a missing cost label from silently dropping work from approval.
    return True


def _provider_argv_option(argv: Any, option: str) -> tuple[bool, Optional[str]]:
    """Return one provider option from an adapter argv template."""
    if not isinstance(argv, list) or option not in argv:
        return False, None
    index = argv.index(option)
    if index + 1 >= len(argv) or not isinstance(argv[index + 1], str):
        return True, None
    value = argv[index + 1].strip()
    return True, value or None


def _canonical_provider_id(value: Any) -> Optional[str]:
    """Normalize provider aliases used by materialized plans and adapter routes."""
    if not isinstance(value, str) or not value.strip():
        return None
    provider_id = value.strip().lower()
    return {
        "fal-serverless": "fal",
        "modal-platform": "modal",
    }.get(provider_id, provider_id)


def _provider_resolution(provider_ids: Iterable[str], source: str) -> dict[str, Any]:
    """Build one deterministic provider resolution for an approval line."""
    normalized = sorted(
        {
            provider_id
            for value in provider_ids
            if (provider_id := _canonical_provider_id(value)) is not None
        }
    )
    return {
        "provider_id": normalized[0] if len(normalized) == 1 else "mixed" if normalized else None,
        "provider_ids": normalized,
        "source": source,
    }


def _provider_environment_value(
    adapter: Mapping[str, Any],
    runtime_environment: Mapping[str, str],
    key: str,
) -> tuple[Optional[str], Optional[str]]:
    """Read one route variable from the adapter overlay, then the runtime environment."""
    adapter_environment = adapter.get("environment")
    if isinstance(adapter_environment, Mapping):
        value = adapter_environment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), f"adapter {adapter.get('adapter_id')} environment.{key}"
    value = runtime_environment.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip(), f"runtime environment {key}"
    return None, None


def _declared_item_provider(
    item: Mapping[str, Any],
    label: str,
) -> Optional[dict[str, Any]]:
    """Read an explicit stage or adapter provider declaration."""
    for key in ("provider_id", "execution_provider"):
        provider_id = _canonical_provider_id(item.get(key))
        if provider_id is not None:
            return _provider_resolution([provider_id], f"{label}.{key}")
    provider = item.get("provider")
    if isinstance(provider, Mapping):
        provider_id = _canonical_provider_id(provider.get("provider_id"))
        if provider_id is not None:
            return _provider_resolution([provider_id], f"{label}.provider.provider_id")
    return None


def _adapter_provider_resolution(
    plan: Mapping[str, Any],
    adapter: Mapping[str, Any],
    runtime_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Resolve one adapter provider from its command and current environment."""
    adapter_id = str(adapter.get("adapter_id", "<unnamed adapter>"))
    declared = _declared_item_provider(adapter, f"adapter {adapter_id}")
    if declared is not None:
        return declared

    adapter_environment = adapter.get("environment")
    if (
        isinstance(adapter_environment, Mapping)
        and adapter_environment.get("CLAUDE_BINDER_EXECUTION_ROUTE") == "modal-platform"
    ):
        return _provider_resolution(
            ["modal"],
            f"adapter {adapter_id} environment.CLAUDE_BINDER_EXECUTION_ROUTE",
        )

    argv = adapter.get("command_argv_template")
    command = argv if isinstance(argv, list) else []
    module_name = None
    if "-m" in command:
        module_index = command.index("-m")
        if module_index + 1 < len(command) and isinstance(command[module_index + 1], str):
            module_name = command[module_index + 1]

    protocol_present, protocol = _provider_argv_option(command, "--runner-protocol")
    if protocol_present and protocol is None:
        return _provider_resolution([], f"adapter {adapter_id} has an empty --runner-protocol")
    if protocol_present and protocol != "auto":
        provider_id = _canonical_provider_id(protocol)
        if provider_id is None:
            return _provider_resolution([], f"adapter {adapter_id} has an invalid --runner-protocol")
        return _provider_resolution(
            [provider_id],
            f"adapter {adapter_id} command_argv_template --runner-protocol",
        )

    fal_url_present, fal_url = _provider_argv_option(command, "--fal-url")
    if fal_url_present:
        if fal_url is None:
            return _provider_resolution([], f"adapter {adapter_id} has an empty --fal-url")
        return _provider_resolution(
            ["fal"],
            f"adapter {adapter_id} command_argv_template --fal-url",
        )
    if isinstance(module_name, str) and module_name.rsplit(".", 1)[-1].startswith("fal_"):
        return _provider_resolution(
            ["fal"],
            f"adapter {adapter_id} command module {module_name}",
        )

    if module_name == "claude_binder.adapters.proteinmpnn_designer":
        fal_environment, fal_source = _provider_environment_value(
            adapter,
            runtime_environment,
            "PROTEINMPNN_FAL_URL",
        )
        if fal_environment is not None:
            return _provider_resolution(["fal"], str(fal_source))
        return _provider_resolution(
            ["local"],
            f"adapter {adapter_id} ProteinMPNN auto route without --fal-url or PROTEINMPNN_FAL_URL",
        )

    if module_name == "claude_binder.adapters.rfdiffusion_generator":
        root_present, root_value = _provider_argv_option(command, "--rfdiffusion-root")
        if root_present and root_value is not None:
            return _provider_resolution(
                ["local"],
                f"adapter {adapter_id} command_argv_template --rfdiffusion-root",
            )
        environment_root, root_source = _provider_environment_value(
            adapter,
            runtime_environment,
            "RFDIFFUSION_ROOT",
        )
        if environment_root is not None:
            return _provider_resolution(["local"], str(root_source))
        _, weights_value = _provider_argv_option(command, "--weights-dir")
        modal_root = Path("/opt/rfd")
        modal_weights = Path(weights_value).expanduser() if weights_value else Path("/weights")
        if modal_root.is_dir() and modal_weights.is_dir():
            return _provider_resolution(
                ["modal"],
                f"adapter {adapter_id} RFdiffusion auto route found /opt/rfd and {modal_weights}",
            )
        return _provider_resolution(
            [],
            f"adapter {adapter_id} RFdiffusion auto route found no RFDIFFUSION_ROOT and no /opt/rfd plus {modal_weights}",
        )

    provider = plan.get("provider")
    if isinstance(provider, Mapping):
        provider_id = _canonical_provider_id(provider.get("provider_id"))
        if provider_id is not None:
            return _provider_resolution([provider_id], "provider.provider_id")
    return _provider_resolution(
        ["local"],
        f"adapter {adapter_id} command has no external provider route",
    )


def _resolve_stage_provider(
    plan: Mapping[str, Any],
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
    *,
    runtime_environment: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Resolve one paid stage provider from its stage, nested adapters, and environment."""
    environment = os.environ if runtime_environment is None else runtime_environment
    stage_id = str(stage.get("stage_id", "<unnamed stage>"))
    declared = _declared_item_provider(stage, f"stage {stage_id}")
    if declared is not None:
        return declared

    if adapter.get("adapter_id") == "control-builder":
        cofold = plan.get("cofold")
        predictors = cofold.get("predictors") if isinstance(cofold, Mapping) else None
        adapter_values = plan.get("adapters")
        adapter_items = adapter_values if isinstance(adapter_values, list) else []
        adapter_map = {
            str(item.get("adapter_id")): item
            for item in adapter_items
            if isinstance(item, Mapping) and isinstance(item.get("adapter_id"), str)
        }
        nested: list[tuple[str, dict[str, Any]]] = []
        if isinstance(predictors, list):
            for predictor in predictors:
                if not isinstance(predictor, Mapping) or predictor.get("enabled", True) is not True:
                    continue
                predictor_adapter_id = predictor.get("adapter_id")
                predictor_adapter = adapter_map.get(str(predictor_adapter_id))
                if predictor_adapter is None:
                    return _provider_resolution(
                        [],
                        f"stage {stage_id} cofold predictor adapter {predictor_adapter_id} is missing",
                    )
                nested.append(
                    (
                        str(predictor_adapter_id),
                        _adapter_provider_resolution(plan, predictor_adapter, environment),
                    )
                )
        if nested:
            unresolved = [adapter_id for adapter_id, item in nested if not item["provider_ids"]]
            if unresolved:
                return _provider_resolution(
                    [],
                    f"stage {stage_id} has unresolved cofold predictor adapters: {', '.join(unresolved)}",
                )
            provider_ids = {
                provider_id
                for _, item in nested
                for provider_id in item["provider_ids"]
            }
            sources = "; ".join(
                f"{adapter_id}: {item['source']}" for adapter_id, item in nested
            )
            return _provider_resolution(
                provider_ids,
                f"stage {stage_id} cofold.predictors resolved through {sources}",
            )

    return _adapter_provider_resolution(plan, adapter, environment)


def _provider_ids_from_record(record: Any) -> list[str]:
    """Return the canonical providers named by one plan or estimate record."""
    if not isinstance(record, Mapping):
        return []
    values = record.get("provider_ids")
    if isinstance(values, list):
        provider_ids = sorted(
            {
                provider_id
                for value in values
                if (provider_id := _canonical_provider_id(value)) is not None
            }
        )
        if provider_ids:
            return provider_ids
    provider_id = _canonical_provider_id(record.get("provider_id"))
    return [] if provider_id in {None, "mixed"} else [provider_id]


def _provider_display(provider_ids: Iterable[str]) -> str:
    """Render provider names for the approval card."""
    display = {
        "modal": "Modal",
        "fal": "fal",
        "runpod": "RunPod",
        "lambda": "Lambda",
    }
    normalized = sorted(
        {
            provider_id
            for value in provider_ids
            if (provider_id := _canonical_provider_id(value)) is not None
        }
    )
    return " + ".join(display.get(provider_id, provider_id) for provider_id in normalized) or "UNRESOLVED"


def _provider_detail(
    plan: Mapping[str, Any],
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the materialized provider record with the current route resolution."""
    stage_id = str(stage.get("stage_id", "<unnamed stage>"))
    records = plan.get("paid_stage_providers")
    recorded = next(
        (
            record
            for record in records
            if isinstance(record, Mapping) and record.get("stage_id") == stage_id
        ),
        None,
    ) if isinstance(records, list) else None
    resolved = _resolve_stage_provider(plan, stage, adapter)
    recorded_ids = _provider_ids_from_record(recorded)
    resolved_ids = _provider_ids_from_record(resolved)
    required = plan.get("paid_stage_provider_schema_version") is not None
    if not required:
        status = "UNRECORDED"
        reason = "the plan predates paid stage provider records"
    elif plan.get("paid_stage_provider_schema_version") != PAID_STAGE_PROVIDER_SCHEMA_VERSION:
        status = "UNRESOLVED"
        reason = "the plan uses an unsupported paid stage provider schema"
    elif not recorded_ids:
        status = "UNRESOLVED"
        reason = "the materialized plan has no resolved provider record for this paid stage"
    elif not resolved_ids:
        status = "UNRESOLVED"
        reason = str(resolved.get("source", "the runtime route is unresolved"))
    elif recorded_ids != resolved_ids:
        status = "MISMATCH"
        reason = (
            f"recorded provider {_provider_display(recorded_ids)} differs from "
            f"resolved provider {_provider_display(resolved_ids)}"
        )
    else:
        status = "MATCH"
        reason = "the current route matches the materialized provider record"
    return {
        "provider_status": status,
        "provider_id": resolved.get("provider_id"),
        "provider_ids": resolved_ids,
        "provider_source": resolved.get("source"),
        "recorded_provider_id": recorded.get("provider_id") if isinstance(recorded, Mapping) else None,
        "recorded_provider_ids": recorded_ids,
        "recorded_provider_source": recorded.get("source") if isinstance(recorded, Mapping) else None,
        "provider_record": dict(recorded) if isinstance(recorded, Mapping) else None,
        "provider_reason": reason,
    }


def _width_from_value(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    if float(value) != number or number < 1:
        return None
    return number


def _optimization_round_width_upper_bound(
    plan: dict[str, Any], stage: dict[str, Any]
) -> tuple[Optional[int], str]:
    """Return the configured children-per-round bound for one optimizer predictor.

    Materialization records the aggregate child bound across every configured
    round. The expanded graph records each round on its predictor stage. Their
    quotient is a conservative per-stage workload limit when the filter's
    artifact width is unavailable before execution.
    """
    stage_id = str(stage.get("stage_id", "<unnamed stage>"))
    if not stage_id.startswith("optimization-cofold-round-"):
        return None, f"{stage_id} has no resolved artifact width"
    counts = plan.get("fanout_estimate", {}).get("counts", {})
    total_children = (
        _width_from_value(counts.get("optimized_variants_upper_bound"))
        if isinstance(counts, dict)
        else None
    )
    if total_children is None:
        return (
            None,
            "optimization workload upper bound requires "
            "plan.fanout_estimate.counts.optimized_variants_upper_bound",
        )
    round_numbers = {
        _width_from_value(candidate.get("optimization_round"))
        for candidate in _stage_list(plan)
        if str(candidate.get("stage_id", "")).startswith("optimization-cofold-round-")
    }
    round_numbers.discard(None)
    round_count = len(round_numbers)
    if round_count < 1:
        return None, "optimization workload upper bound requires materialized optimization round indexes"
    if total_children % round_count != 0:
        return (
            None,
            "optimization workload upper bound requires "
            "optimized_variants_upper_bound divisible by the materialized round count",
        )
    children_per_round = total_children // round_count
    if children_per_round < 1:
        return None, "optimization workload upper bound requires at least one configured child per round"
    return (
        children_per_round,
        "upper bound from plan.fanout_estimate.counts.optimized_variants_upper_bound "
        f"({total_children}) divided by {round_count} materialized optimization rounds",
    )


def _resolved_stage_width(
    plan: dict[str, Any], stage: dict[str, Any]
) -> tuple[Optional[int], str, str]:
    stage_id = str(stage.get("stage_id", "<unnamed stage>"))
    for key in ("resolved_width", "resolved_artifact_width", "artifact_width", "width"):
        width = _width_from_value(stage.get(key))
        if width is not None:
            return width, f"stage.{key}", "RESOLVED"
    for key in ("resolved_fanout", "fanout_resolution", "fanout_resolved", "resolved_widths", "resolved_artifact_widths"):
        value = _lookup_stage_map(plan, (key,), stage_id)
        if isinstance(value, dict):
            value = value.get("resolved_width", value.get("artifact_width", value.get("width", value.get("count"))))
        width = _width_from_value(value)
        if width is not None:
            return width, f"plan.{key}[{stage_id}]", "RESOLVED"
    fanout = stage.get("fanout")
    if isinstance(fanout, dict):
        width = _width_from_value(fanout.get("resolved_width"))
        if width is not None:
            return width, "stage.fanout.resolved_width", "RESOLVED"
        width = _width_from_value(fanout.get("scale_count"))
        if width is not None:
            return width, "stage.fanout.scale_count", "RESOLVED"
        source = fanout.get("count_from")
        if isinstance(source, dict):
            for key in ("resolved_width", "artifact_width", "width", "count", "records"):
                width = _width_from_value(source.get(key))
                if width is not None:
                    return width, f"stage.fanout.count_from.{key}", "RESOLVED"
            source_key = f"{source.get('stage_id')}:{source.get('artifact_id')}"
            for key in ("resolved_artifact_widths", "artifact_widths", "resolved_fanout"):
                value = _lookup_stage_map(plan, (key,), source_key)
                width = _width_from_value(value)
                if width is not None:
                    return width, f"plan.{key}[{source_key}]", "RESOLVED"
            source_stage_id = source.get("stage_id")
            source_stage = next(
                (
                    candidate
                    for candidate in _stage_list(plan)
                    if candidate.get("stage_id") == source_stage_id
                ),
                None,
            )
            if isinstance(source_stage, dict):
                source_fanout = source_stage.get("fanout")
                source_width = (
                    _width_from_value(source_fanout.get("scale_count"))
                    if isinstance(source_fanout, dict)
                    else None
                )
                source_outputs = source_stage.get("outputs")
                source_output = (
                    source_outputs[0]
                    if isinstance(source_outputs, list) and source_outputs and isinstance(source_outputs[0], dict)
                    else {}
                )
                source_records = _width_from_value(source_output.get("records_per_count"))
                if source_width is not None:
                    if source.get("value") == "records" and source_records is not None:
                        return (
                            source_width * source_records,
                            f"stage {source_stage_id} fanout.scale_count and records_per_count",
                            "RESOLVED",
                        )
                    if source.get("value") != "records":
                        return source_width, f"stage {source_stage_id} fanout.scale_count", "RESOLVED"
    estimate_counts = plan.get("fanout_estimate", {}).get("counts", {})
    if isinstance(estimate_counts, dict):
        stage_id = str(stage.get("stage_id", ""))
        records_per_count = _width_from_value(
            (
                stage.get("outputs", [{}])[0].get("records_per_count", 1)
                if isinstance(stage.get("outputs"), list) and stage.get("outputs")
                and isinstance(stage["outputs"][0], dict)
                else 1
            )
        )
        matched_stage_ids = [
            str(candidate.get("stage_id", ""))
            for candidate in _stage_list(plan)
            if (
                (stage_id.startswith("cofold-rescore-") and str(candidate.get("stage_id", "")).startswith("cofold-rescore-"))
                or (
                    stage_id.startswith("cofold-screen-")
                    and str(candidate.get("stage_id", "")).startswith("cofold-screen-")
                )
            )
        ]
        if len(matched_stage_ids) == 1 and records_per_count is not None:
            bucket = (
                "rescore_predictions"
                if stage_id.startswith("cofold-rescore-")
                else "screen_predictions"
                if stage_id.startswith("cofold-screen-")
                else None
            )
            totals = estimate_counts.get("provider_calls_including_smokes")
            smoke_by_stage = estimate_counts.get("smoke_calls_by_stage")
            if (
                bucket is not None
                and isinstance(totals, dict)
                and isinstance(smoke_by_stage, dict)
            ):
                total = _width_from_value(totals.get(bucket))
                smoke = _width_from_value(smoke_by_stage.get(stage_id))
                if total is not None and smoke is not None and total >= smoke:
                    scale_calls = total - smoke
                    if scale_calls % records_per_count == 0:
                        return (
                            scale_calls // records_per_count,
                            f"plan.fanout_estimate.counts.provider_calls_including_smokes.{bucket}",
                            "RESOLVED",
                        )
    upper_bound, upper_bound_source = _optimization_round_width_upper_bound(plan, stage)
    if upper_bound is not None:
        return upper_bound, upper_bound_source, "UPPER_BOUND"
    if stage_id.startswith("optimization-cofold-round-"):
        return None, upper_bound_source, "UNKNOWN"
    if stage.get("mode") == "single":
        return 1, "stage.mode=single", "RESOLVED"
    return None, f"{stage_id} has no resolved artifact width", "UNKNOWN"


def _timing_basis(
    plan: dict[str, Any],
    stage: dict[str, Any],
    adapter: dict[str, Any],
    providers: Iterable[str],
) -> tuple[Any, str]:
    """Return the timing basis for one stage, keyed on the provider it resolves to.

    The three built-in bases below were measured on fal, so they belong to a
    stage that resolves to fal. Keying them on the campaign tier label instead
    made a mixed campaign price its Modal stage from a fal H100 measurement when
    the label read fal, and drop the fal stages' known timings when it read
    modal. The tier label is one word for a whole graph; the route is per stage.
    """
    stage_id = str(stage.get("stage_id", "<unnamed stage>"))
    on_fal = "fal" in set(providers)
    for item, source in (
        (stage.get("timing_basis"), f"stage {stage_id} timing_basis"),
        (stage.get("timing"), f"stage {stage_id} timing"),
        (stage.get("estimate"), f"stage {stage_id} estimate"),
        (_lookup_stage_map(plan, ("timing_basis", "stage_timing", "stage_timings"), stage_id),
         f"plan timing for {stage_id}"),
        (adapter.get("timing_basis"), f"adapter {adapter.get('adapter_id')} timing_basis"),
        (adapter.get("timing"), f"adapter {adapter.get('adapter_id')} timing"),
    ):
        if isinstance(item, dict):
            return item, source
    adapter_id = adapter.get("adapter_id")
    if on_fal and adapter_id == "rfdiffusion-generator":
        return {
            "startup_seconds": 0,
            "item_seconds": RFDIFFUSION3_BILLED_SECONDS_PER_BACKBONE,
            "machine": "GPU-H100",
            "measurement_unit": "backbone",
        }, SOURCE_TIMINGS
    if on_fal and adapter_id == "proteinmpnn-designer":
        return {
            "startup_seconds": 0,
            "item_seconds": PROTEINMPNN_BILLED_SECONDS_PER_DESIGN_CALL,
            "machine": "M",
            "measurement_unit": "design call",
            "calls_per_count": 1,
        }, SOURCE_TIMINGS
    if on_fal and adapter_id == "esmfold2-fast-predictor":
        return {
            "startup_seconds": 0,
            "item_seconds": {
                "minimum": ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MIN,
                "maximum": ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MAX,
            },
            "machine": "GPU-H100",
            "measurement_unit": "fold",
        }, SOURCE_TIMINGS
    return None, "no recorded timing basis"


def _timing_component(basis: dict[str, Any], keys: tuple[str, ...]) -> tuple[Optional[float], Optional[float]]:
    for key in keys:
        if key in basis:
            bounds = _bounds(basis[key])
            if bounds[0] is not None and bounds[1] is not None:
                return bounds
        flat_bounds = _bounds({
            "minimum": basis.get(f"{key}_min"),
            "maximum": basis.get(f"{key}_max"),
        })
        if flat_bounds[0] is not None and flat_bounds[1] is not None:
            return flat_bounds
    return None, None


DEFAULT_SCALAR_RATE_MACHINE = "GPU-H100"
"""Machine the bare ``rate_usd_per_gpu_second`` scalar prices when no caller names one.

The scalar is a single number for a whole graph, so it needs a machine before it can be
compared against a stage's own label. A caller reading an operator rate record passes the
machine it read the scalar from. This default applies when the caller names none.
"""

KNOWN_PROVIDER_IDS = ("fal", "modal", "runpod")
"""Provider ids that may appear as the leading token of a machine label."""


def _resolved_provider_ids(provider_ids: Iterable[Any]) -> list[str]:
    found: list[str] = []
    for value in provider_ids or ():
        provider_id = _canonical_provider_id(value)
        if provider_id is not None and provider_id not in found:
            found.append(provider_id)
    return found


def rate_lookup_keys(provider_ids: Iterable[Any], machine: Any) -> tuple[str, ...]:
    """Rate-record keys for one stage's machine, most specific first.

    A machine label names hardware, and two providers bill the same hardware at different
    prices, so a bare label cannot select a rate on its own. The provider qualifies the
    lookup. It is read from the plan, where the route is resolved per stage, rather than
    from the machine string, where it would be hand-written and could disagree with the
    route the stage actually takes.

    When the label does carry a provider and that provider disagrees with the resolved
    route, no key is returned and the stage goes unpriced. Twenty-one fal adapters in
    shipped plans have quoted a Modal machine, and falling back to such a label would price
    a fal stage from a Modal rate. Refusing an ambiguous label costs an estimate. Trusting
    it costs money.

    A rate record that prices each provider separately matches the qualified key, and one
    that prices hardware alone matches the bare label. Both are returned, most specific
    first.
    """
    if not isinstance(machine, str) or not machine:
        return ()
    resolved = _resolved_provider_ids(provider_ids)
    label_provider, hardware = split_machine_label(machine)
    if label_provider is not None and resolved and label_provider not in resolved:
        return ()
    keys: list[str] = []
    for provider_id in resolved:
        qualified = f"{provider_id} {hardware or machine}"
        if qualified not in keys:
            keys.append(qualified)
    for candidate in (machine, hardware):
        if isinstance(candidate, str) and candidate and candidate not in keys:
            keys.append(candidate)
    return tuple(keys)


def scalar_rate_machines(provider_ids: Iterable[Any], scalar_machine: Any) -> tuple[str, ...]:
    """Machine labels a scalar rate measured on ``scalar_machine`` is allowed to price.

    A profile may spell the hardware bare or already qualified by its provider, and so may
    the caller naming the scalar's machine. Both spellings are reduced to hardware first, so
    the comparison is symmetric whichever side carries the provider.
    """
    if not isinstance(scalar_machine, str) or not scalar_machine:
        return ()
    _label_provider, hardware = split_machine_label(scalar_machine)
    return rate_lookup_keys(provider_ids, hardware or scalar_machine)


def split_machine_label(machine: Any) -> tuple[Optional[str], Optional[str]]:
    """Split a machine label into its provider and its hardware.

    ``"fal GPU-H100"`` splits. ``"GPU-H100"`` has no provider and is returned whole. The
    split exists so two derivations of one unit price can be compared when one label
    carries its provider and the other does not.
    """
    if not isinstance(machine, str) or not machine.strip():
        return None, None
    label = machine.strip()
    head, separator, tail = label.partition(" ")
    provider_id = _canonical_provider_id(head.rstrip(","))
    if separator and provider_id in KNOWN_PROVIDER_IDS and tail.strip():
        return provider_id, tail.strip()
    return None, label


@dataclass(frozen=True)
class UnitPrice:
    """One derivation of the price of a single billed unit.

    The package derives a unit price twice by two routes that never met. A stage's
    ``timing_basis`` bounds the seconds a unit bills and is multiplied by an operator rate
    to gate campaign dispatch. An adapter's ``qualification.cost_basis`` carries a dollar
    amount directly and gates canary qualification. Both describe one fold on one machine,
    so a large disagreement between them is a fact about the package worth reporting.
    """

    adapter_id: str
    derivation: str
    provider_id: Optional[str]
    hardware: Optional[str]
    measurement_unit: Optional[str]
    usd_per_unit_max: Optional[float]
    source: str


def unit_price_derivations(
    profile: Mapping[str, Any],
    *,
    rates_usd_per_machine_second: Mapping[str, float],
) -> list[UnitPrice]:
    """Collect every per-unit price a resolved profile derives, by either route.

    Pass a profile resolved through ``lane.load_profile``. Reading a template file
    directly misses every adapter an overlay inherits rather than declares.

    Only the upper bound is collected. The ceiling is what gates spend, so the ceiling is
    what two derivations must agree about.
    """
    found: list[UnitPrice] = []

    def rate_for(machine: Any, provider_id: Optional[str], hardware: Optional[str]) -> Optional[float]:
        for key in rate_lookup_keys([provider_id] if provider_id else [], machine):
            if key in rates_usd_per_machine_second:
                return _finite_bound(rates_usd_per_machine_second[key])
        if hardware and hardware in rates_usd_per_machine_second:
            return _finite_bound(rates_usd_per_machine_second[hardware])
        return None

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            adapter_id = node.get("adapter_id")
            if isinstance(adapter_id, str) and adapter_id:
                timing = node.get("timing_basis")
                if isinstance(timing, Mapping):
                    provider_id, hardware = split_machine_label(timing.get("machine"))
                    _, item_max = _timing_component(
                        dict(timing),
                        ("item_seconds", "fold_seconds", "per_item_seconds",
                         "per_prediction_seconds", "item", "fold", "gpu_seconds_per_item"),
                    )
                    rate = rate_for(timing.get("machine"), provider_id, hardware)
                    found.append(UnitPrice(
                        adapter_id=adapter_id,
                        derivation="timing_basis",
                        provider_id=provider_id,
                        hardware=hardware,
                        measurement_unit=timing.get("measurement_unit"),
                        usd_per_unit_max=(
                            item_max * rate if item_max is not None and rate is not None else None
                        ),
                        source=str(timing.get("source", "")),
                    ))
                cost = (node.get("qualification") or {}).get("cost_basis") if isinstance(
                    node.get("qualification"), Mapping) else None
                if isinstance(cost, Mapping):
                    provider_id, hardware = split_machine_label(cost.get("machine"))
                    amount = _finite_bound(cost.get("amount_usd"))
                    count = cost.get("unit_count")
                    per_unit = (
                        amount / count
                        if amount is not None and isinstance(count, (int, float))
                        and not isinstance(count, bool) and count > 0
                        else None
                    )
                    found.append(UnitPrice(
                        adapter_id=adapter_id,
                        derivation="qualification.cost_basis",
                        provider_id=provider_id,
                        hardware=hardware,
                        measurement_unit=cost.get("measurement_unit"),
                        usd_per_unit_max=per_unit,
                        source=str(cost.get("source", "")),
                    ))
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(profile)
    return found


def compare_unit_prices(
    derivations: Iterable[UnitPrice],
    *,
    tolerance: float = 1.05,
) -> list[dict[str, Any]]:
    """Report groups where two derivations of one unit price disagree.

    Grouped on hardware and measurement unit rather than on the full machine label,
    because one derivation may spell its provider and the other may not. That spelling
    difference must not hide the comparison.

    A ratio at or under ``tolerance`` is agreement. Anything above is returned for a human
    to explain or fix. Two honest measurements of different quantities can land here, so a
    result is a prompt to say which quantity each one measures, not proof of an error.
    """
    groups: dict[tuple[Any, Any], list[UnitPrice]] = {}
    for item in derivations:
        if item.usd_per_unit_max is None or not item.measurement_unit:
            continue
        groups.setdefault((item.hardware, item.measurement_unit), []).append(item)
    disagreements: list[dict[str, Any]] = []
    for (hardware, unit), members in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if len(members) < 2:
            continue
        prices = [m.usd_per_unit_max for m in members]
        low, high = min(prices), max(prices)
        if low <= 0:
            continue
        ratio = high / low
        if ratio <= tolerance:
            continue
        disagreements.append({
            "hardware": hardware,
            "measurement_unit": unit,
            "ratio": ratio,
            "low_usd_per_unit": low,
            "high_usd_per_unit": high,
            "members": [
                {
                    "adapter_id": m.adapter_id,
                    "derivation": m.derivation,
                    "provider_id": m.provider_id,
                    "usd_per_unit_max": m.usd_per_unit_max,
                }
                for m in sorted(members, key=lambda m: (m.adapter_id, m.derivation))
            ],
        })
    return disagreements


def estimate_from_graph(
    plan: dict[str, Any] | Path,
    tier: str = "modal",
    *,
    rate_usd_per_gpu_second: Optional[float] = None,
    rate_source: Optional[str] = None,
    rates_usd_per_machine_second: Optional[dict[str, float]] = None,
    rate_machine: Optional[str] = None,
) -> Estimate:
    """Estimate every paid stage in a materialized execution graph.

    A missing width or timing upper bound makes the whole estimate UNKNOWN.
    An optimization predictor whose width resolves from a later artifact uses
    the materialized children-per-round upper bound and records that fact.
    This function accepts no ceiling. A ceiling is a user input checked by
    approve(), never an estimator default.
    """
    if isinstance(plan, Path):
        plan = json.loads(plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("materialized plan must be an object")
    if tier not in {"fal", "modal", "runpod", "lambda"}:
        raise ValueError(f"unsupported provider tier {tier!r}")
    design_count = plan_final_count(plan)

    stages = _stage_list(plan)
    stage_estimates: list[dict[str, Any]] = []
    covered: list[str] = []
    upper_bound_stages: list[str] = []
    unknown: list[str] = []
    warnings: list[str] = []
    total_min = 0.0
    total_max = 0.0
    workload_units = 0
    all_bounded = True
    for stage in stages:
        stage_id = str(stage.get("stage_id", "<unnamed stage>"))
        adapter = _stage_adapter(plan, stage)
        paid = _stage_is_paid(stage, adapter)
        width, width_source, width_kind = _resolved_stage_width(plan, stage)
        outputs = stage.get("outputs")
        first_output = outputs[0] if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) else {}
        # stage_provider_calls defaults a missing records_per_count to one
        # provider record per invocation.
        records_per_count = first_output.get("records_per_count", 1)
        if isinstance(records_per_count, bool) or not isinstance(records_per_count, int) or records_per_count < 1:
            records_per_count = None
        # execute_stage runs smoke and scale for smoke_scale, and one single
        # phase for single.
        batches = 2 if stage.get("mode") == "smoke_scale" else 1
        if width is None or records_per_count is None:
            workload = None
        elif stage.get("mode") == "smoke_scale":
            # lane.py runs one smoke batch and one scale batch for this mode.
            workload = records_per_count * (width + 1)
        else:
            workload = records_per_count * width
        detail: dict[str, Any] = {
            "stage_id": stage_id,
            "paid": paid,
            "resolved_width": width,
            "width_source": width_source,
            "width_kind": width_kind,
            "records_per_count": records_per_count,
            "batches": batches,
            "workload_units": workload,
        }
        if not paid:
            detail["status"] = "FREE"
            stage_estimates.append(detail)
            continue
        detail.update(_provider_detail(plan, stage, adapter))
        basis, basis_source = _timing_basis(
            plan, stage, adapter, detail.get("provider_ids") or ()
        )
        if isinstance(basis, dict) and basis.get("calls_per_count") == 1 and width is not None:
            workload = width + 1 if stage.get("mode") == "smoke_scale" else width
            detail["workload_units"] = workload
            detail["workload_source"] = "one provider call per stage count"
        if workload is not None:
            workload_units += workload
        if width is None:
            reason = width_source
            required_value = "a resolved stage width or a declared maximum workload"
        elif records_per_count is None:
            reason = "records_per_count must be a positive integer"
            required_value = "outputs[0].records_per_count as a positive integer"
        elif not isinstance(basis, dict):
            reason = basis_source
            required_value = "a timing_basis with billed startup and per-item upper bounds"
        else:
            startup_min, startup_max = _timing_component(
                basis,
                ("startup_seconds", "load_seconds", "startup", "load", "gpu_seconds_per_batch"),
            )
            item_min, item_max = _timing_component(
                basis,
                ("item_seconds", "fold_seconds", "per_item_seconds", "per_prediction_seconds",
                 "item", "fold", "gpu_seconds_per_item"),
            )
            if workload is None:
                reason = "stage workload is unresolved"
                required_value = "a resolved stage width or a declared maximum workload"
            elif startup_min is None or item_min is None:
                missing_components = []
                if startup_min is None:
                    missing_components.append("startup_seconds")
                if item_min is None:
                    missing_components.append("item_seconds")
                reason = f"{basis_source} lacks upper bounds for " + ", ".join(missing_components)
                required_value = "upper bounds for " + ", ".join(missing_components)
            else:
                gpu_min = batches * startup_min + workload * item_min
                gpu_max = batches * startup_max + workload * item_max
                detail.update({
                    "status": "BOUNDED",
                    "timing_source": basis_source,
                    "machine": basis.get("machine"),
                    "measurement_unit": basis.get("measurement_unit"),
                    "gpu_seconds_min": gpu_min,
                    "gpu_seconds_max": gpu_max,
                })
                total_min += gpu_min
                total_max += gpu_max
                covered.append(stage_id)
                if width_kind == "UPPER_BOUND":
                    upper_bound_stages.append(stage_id)
                stage_estimates.append(detail)
                continue
        detail.update({
            "status": "UNKNOWN",
            "reason": reason,
            "required_value": required_value,
        })
        unknown.append(stage_id)
        all_bounded = False
        stage_estimates.append(detail)

    if unknown:
        warnings.append("paid stages without a bound: " + ", ".join(unknown))
    if upper_bound_stages:
        warnings.append(
            "upper-bound workloads used for paid stages: " + ", ".join(upper_bound_stages)
        )
    if not all_bounded:
        total_min_value: Optional[float] = None
        total_max_value: Optional[float] = None
    else:
        total_min_value = total_min
        total_max_value = total_max
    usd_min = None
    usd_max = None
    status = "UNKNOWN"
    rate_map: dict[str, float] = {}
    if rates_usd_per_machine_second is not None:
        for machine, rate in rates_usd_per_machine_second.items():
            normalized_rate = _finite_bound(rate)
            if (
                not isinstance(machine, str)
                or not machine
                or normalized_rate is None
                or normalized_rate == 0
            ):
                raise ValueError("rates_usd_per_machine_second must map machine names to positive finite rates")
            rate_map[machine] = normalized_rate
    scalar_rate = _finite_bound(rate_usd_per_gpu_second)
    scalar_machine = rate_machine if isinstance(rate_machine, str) and rate_machine else DEFAULT_SCALAR_RATE_MACHINE
    rate_provided = scalar_rate is not None or bool(rate_map)
    if rate_provided and not isinstance(rate_source, str):
        warnings.append("price is UNKNOWN because the supplied rate has no source string")
    elif rate_provided and not rate_source.strip():
        warnings.append("price is UNKNOWN because the supplied rate source is blank")
    elif total_min_value is not None and rate_provided:
        priced_min = 0.0
        priced_max = 0.0
        missing_rate_stages: list[str] = []
        for detail in stage_estimates:
            if detail.get("status") != "BOUNDED":
                continue
            machine = detail.get("machine")
            provider_ids = detail.get("provider_ids") or ()
            candidate_keys = rate_lookup_keys(provider_ids, machine)
            stage_rate = None
            rate_key = None
            for key in candidate_keys:
                if key in rate_map:
                    stage_rate = rate_map[key]
                    rate_key = key
                    break
            label_provider, stage_hardware = split_machine_label(machine)
            resolved_providers = _resolved_provider_ids(provider_ids)
            # A label naming a provider the stage does not route to is not usable as a rate
            # key and is not usable as a scalar match either. A basis carrying no machine at
            # all is a different case: nothing contradicts the scalar, so it still applies.
            label_conflicts = (
                label_provider is not None
                and bool(resolved_providers)
                and label_provider not in resolved_providers
            )
            scalar_machines = scalar_rate_machines(provider_ids, scalar_machine)
            if stage_rate is None and not label_conflicts and (
                machine is None
                or machine in scalar_machines
                or (stage_hardware is not None and stage_hardware in scalar_machines)
            ):
                stage_rate = scalar_rate
                rate_key = scalar_machine if machine is not None else None
            if stage_rate is None:
                missing_rate_stages.append(str(detail["stage_id"]))
                detail["rate_keys_tried"] = list(candidate_keys)
                continue
            priced_min += float(detail["gpu_seconds_min"]) * stage_rate
            priced_max += float(detail["gpu_seconds_max"]) * stage_rate
            detail["rate_usd_per_second"] = stage_rate
            detail["rate_key"] = rate_key
            detail["rate_keys_tried"] = list(candidate_keys)
            detail["usd_estimate_min"] = float(detail["gpu_seconds_min"]) * stage_rate
            detail["usd_estimate_max"] = float(detail["gpu_seconds_max"]) * stage_rate
        if missing_rate_stages:
            tried = sorted({
                key
                for detail in stage_estimates
                if str(detail.get("stage_id")) in set(missing_rate_stages)
                for key in detail.get("rate_keys_tried", ())
            })
            warnings.append(
                "price unavailable for machine rates used by stages: "
                + ", ".join(missing_rate_stages)
                + (
                    "; the rate record declares none of " + ", ".join(tried)
                    if tried
                    else ""
                )
            )
        else:
            usd_min = priced_min
            usd_max = priced_max
            status = "PARAMETER (read this run)"
    return Estimate(
        designs=design_count,
        seeds_per_design=0,
        folds=workload_units,
        shard_size=None,
        containers=None,
        gpu_seconds_min=total_min_value,
        gpu_seconds_max=total_max_value,
        server_seconds_min=None,
        server_seconds_max=None,
        seconds_kind=(
            "materialized paid-stage graph; every paid stage has a billed-time bound"
            + (
                "; workload upper bounds apply to " + ", ".join(upper_bound_stages)
                if upper_bound_stages
                else ""
            )
            if all_bounded else "UNKNOWN: materialized graph has paid stages without a timing bound"
        ),
        tier=tier,
        usd_estimate_min=usd_min,
        usd_estimate_max=usd_max,
        rate_status=status,
        rate_source=rate_source,
        warnings=warnings,
        stage_count=len(stages),
        paid_stage_count=sum(1 for detail in stage_estimates if detail["paid"]),
        covered_stage_ids=covered,
        upper_bound_stage_ids=upper_bound_stages,
        unknown_stage_ids=unknown,
        stage_estimates=stage_estimates,
        workload_units=workload_units,
        provider_records_required=(
            plan.get("paid_stage_provider_schema_version") is not None
        ),
    )


# Names used by callers that describe the same materialized-plan operation.
estimate_materialized_plan = estimate_from_graph
estimate_plan = estimate_from_graph


def estimate_run(
    designs: int | dict[str, Any] | Path,
    seeds_per_design: int | str | None = None,
    tier: Optional[str] = None,
    *,
    rate_usd_per_gpu_second: Optional[float] = None,
    rate_source: Optional[str] = None,
    rates_usd_per_machine_second: Optional[dict[str, float]] = None,
    modal_command_budget_seconds: Optional[float] = None,
) -> Estimate:
    """Compatibility wrapper for the pre-graph scalar API.

    New callers pass a materialized plan. The integer form remains only for
    existing ledger and self-test readers while they migrate to the graph API.
    """
    if isinstance(designs, (dict, Path)):
        graph_tier = tier or (seeds_per_design if isinstance(seeds_per_design, str) else "modal")
        return estimate_from_graph(
            designs,
            graph_tier,
            rate_usd_per_gpu_second=rate_usd_per_gpu_second,
            rate_source=rate_source,
            rates_usd_per_machine_second=rates_usd_per_machine_second,
        )
    if seeds_per_design is None or tier is None:
        raise TypeError("legacy estimate_run requires designs, seeds_per_design, and tier")
    seeds_per_design = int(seeds_per_design)
    # Legacy compatibility arithmetic. Production approval uses estimate_from_graph.
    # The scalar path represents a fold-only ESMFold2-Fast screen. Its billed
    # range is the measured fal fold range declared above.
    folds = designs * seeds_per_design
    warnings: list[str] = []
    if tier == "fal":
        shard = None
        containers = None
        gpu_seconds_min = folds * ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MIN
        gpu_seconds_max = folds * ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MAX
        server_seconds_min = None
        server_seconds_max = None
        seconds_kind = (
            "MEASURED billed ESMFold2-Fast fold range, DERIVED total: GPU "
            f"{gpu_seconds_min:.3f}-{gpu_seconds_max:.3f} s"
        )
        warnings.append(
            "Fal bills the measured ESMFold2-Fast fold range, including setup and drain "
            f"({SOURCE_TIMINGS})."
        )
    elif tier in {"modal", "runpod", "lambda"}:
        shard = None
        containers = None
        gpu_seconds_min = None
        gpu_seconds_max = None
        server_seconds_min = None
        server_seconds_max = None
        provider_name = {
            "modal": "Modal",
            "runpod": "RunPod",
            "lambda": "Lambda Cloud",
        }[tier]
        seconds_kind = f"UNKNOWN {provider_name} timing shape and range"
        warnings.append(
            f"{provider_name} timing is UNKNOWN: no {provider_name} predictor receipt "
            "exists for this scalar compatibility path. "
            "The fal range is not reused."
        )
        warnings.append(
            f"Measure one {provider_name} smoke job and record image, weight mount, "
            f"per-call load, fold, total, and runner reuse before estimating {provider_name} cost."
        )
    else:
        raise ValueError(f"unsupported provider tier {tier!r}")

    usd_min: Optional[float] = None
    usd_max: Optional[float] = None
    status = "UNKNOWN"
    if (
        rate_usd_per_gpu_second is not None
        and isinstance(rate_source, str)
        and rate_source.strip()
        and gpu_seconds_min is not None
    ):
        usd_min = gpu_seconds_min * rate_usd_per_gpu_second
        usd_max = gpu_seconds_max * rate_usd_per_gpu_second
        status = "PARAMETER (read this run)"
    elif rate_usd_per_gpu_second is not None:
        warnings.append("price is UNKNOWN because the supplied rate has no source string")
    return Estimate(
        designs=designs,
        seeds_per_design=seeds_per_design,
        folds=folds,
        shard_size=shard,
        containers=containers,
        gpu_seconds_min=gpu_seconds_min,
        gpu_seconds_max=gpu_seconds_max,
        server_seconds_min=server_seconds_min,
        server_seconds_max=server_seconds_max,
        seconds_kind=seconds_kind,
        tier=tier,
        usd_estimate_min=usd_min,
        usd_estimate_max=usd_max,
        rate_status=status,
        rate_source=rate_source,
        warnings=list(dict.fromkeys(warnings)),
    )


def largest_fitting_n(
    seeds_per_design: int,
    tier: str,
    ceiling_usd: float,
    rate_usd_per_gpu_second: float,
    spent_so_far_usd: float = 0.0,
) -> Optional[int]:
    """Largest N whose maximum estimated cost fits the remaining ceiling.

    The maximum bound reserves against the observed timing spread. Cost rises
    with N, so fitting sizes are downward closed. Equality passes, matching
    the executor's estimated-greater-than-maximum rejection rule.
    """
    remaining = ceiling_usd - spent_so_far_usd
    n = 1
    best: Optional[int] = None
    while True:
        est = estimate_run(n, seeds_per_design, tier, rate_usd_per_gpu_second=rate_usd_per_gpu_second)
        if est.usd_estimate_max is None:
            return None
        if est.usd_estimate_max > remaining:
            break
        best = n
        n += 1
        if n > 100000:  # pragma: no cover
            break
    return best


def budget_refusal_text(
    est: Estimate,
    ceiling_usd: float,
    spent_so_far_usd: float = 0.0,
    *,
    spent_kind: str = "estimated",
) -> str:
    """The refusal block from the estimator design: estimate, ceiling, largest N that fits."""
    remaining = ceiling_usd - spent_so_far_usd
    if est.gpu_seconds_min is None or est.gpu_seconds_max is None:
        estimate_line = "estimate      UNKNOWN provider timing"
    else:
        estimate_line = (
            f"estimate      {est.gpu_seconds_min:.1f} to "
            f"{est.gpu_seconds_max:.1f} GPU seconds"
        )
    lines = [
        "REFUSED: the estimate exceeds the ceiling.",
        "",
        estimate_line,
        "              one measured per-call range; runner count is UNKNOWN",
        "              " + est.seconds_kind,
    ]
    if est.usd_estimate_min is not None and est.usd_estimate_max is not None and est.rate_source:
        lines.append(
            f"              = estimated total {est.usd_estimate_min:.2f} to "
            f"{est.usd_estimate_max:.2f} USD "
            f"at a rate read from {est.rate_source}"
        )
    lines += [
        f"ceiling       {ceiling_usd:.2f} USD ({'user stated'}); equality passes, exceeding refuses",
        f"remaining     {remaining:.2f} USD after {spent_so_far_usd:.2f} USD "
        f"{spent_kind} spend",
    ]
    largest = None
    if est.usd_estimate_max is not None and est.gpu_seconds_max:
        rate = est.usd_estimate_max / est.gpu_seconds_max
        if rate:
            largest = largest_fitting_n(est.seeds_per_design, est.tier, ceiling_usd, rate, spent_so_far_usd)
    if largest:
        fit = estimate_run(largest, est.seeds_per_design, est.tier)
        lines += [
            f"largest N     {largest} designs x {est.seeds_per_design} seeds = "
            f"{fit.folds} calls, {fit.gpu_seconds_min:.1f} to "
            f"{fit.gpu_seconds_max:.1f} GPU seconds",
        ]
    else:
        lines += ["largest N     none: even the smallest run exceeds the remaining ceiling"]
    lines += [
        "",
        "options       rerun with a smaller explicit scope; the tool proposes one and waits for you",
        "              raising the campaign cap above the account policy cap takes an entry",
        "              in the account holder's account policy record, never a setting here",
        "timings used  billed ESMFold2-Fast fold range "
        f"{ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MIN}-{ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MAX} "
        f"s per call [{SOURCE_TIMINGS}]",
    ]
    return "\n".join(lines)


def price_unavailable_refusal_text(
    tier: str,
    policy: Optional[AccountPolicy] = None,
) -> str:
    """Refusal when no rate parameter exists, so the ceiling cannot be checked."""
    account_policy = policy or default_account_policy()
    return "\n".join([
        "REFUSED: cannot check the spend ceiling.",
        "",
        "reason        no provider rate parameter was supplied this run",
        f"policy        prices are runtime parameters, never constants ({SOURCE_PRICE_WARNING})",
        "consequence   paid approval is blocked until a rate is read from a real price record",
        "options       supply --rate-record with operator-attested planning rates, source, and read_at, then approve",
        f"policy caps   {account_policy.cap_summary()}, from {account_policy.source}",
        f"tier          {tier}",
    ])


# ---------------------------------------------------------------------------
# Approval card and gate. Nothing spends before approve() returns a record.
# ---------------------------------------------------------------------------


def _approval_provider_lines(est: Estimate) -> list[str]:
    """Render one checkable provider line for each estimated paid stage."""
    lines: list[str] = []
    for detail in est.stage_estimates:
        if detail.get("paid") is not True:
            continue
        stage_id = str(detail.get("stage_id", "<unnamed stage>"))
        status = str(detail.get("provider_status", "UNRESOLVED"))
        recorded_ids = detail.get("recorded_provider_ids")
        resolved_ids = detail.get("provider_ids")
        recorded = _provider_display(recorded_ids if isinstance(recorded_ids, list) else [])
        resolved = _provider_display(resolved_ids if isinstance(resolved_ids, list) else [])
        if status == "MATCH":
            source = detail.get("recorded_provider_source") or detail.get("provider_source")
            lines.append(f"provider       {stage_id}: {resolved} (RESOLVED; {source})")
        elif status == "MISMATCH":
            lines.append(
                f"provider       {stage_id}: recorded {recorded}; resolved {resolved} (MISMATCH)"
            )
        elif status == "UNRECORDED":
            lines.append(
                f"provider       {stage_id}: {resolved} (UNRECORDED; materialize a provider record)"
            )
        else:
            lines.append(
                f"provider       {stage_id}: recorded {recorded}; resolved {resolved} "
                f"(UNRESOLVED; {detail.get('provider_reason')})"
            )
    return lines


def materialized_estimate_reconciliation(
    plan: Mapping[str, Any],
    est: Estimate,
    *,
    absolute_tolerance_usd: float = 0.01,
    relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Compare the profile's planning figure with the graph-derived estimate.

    The profile figure is useful provenance, but it cannot be the dispatch basis:
    stage counts and shard widths are resolved only in the materialized graph. A
    harmless stale figure must not block work already protected by the hard cap.
    The approval card therefore shows the disagreement and uses the graph estimate.
    """
    provider = plan.get("provider")
    provider_object = provider if isinstance(provider, Mapping) else {}
    budget = provider_object.get("budget")
    budget_object = budget if isinstance(budget, Mapping) else {}
    profile_value = budget_object.get("estimated_spend_usd")
    profile_estimate = (
        float(profile_value)
        if isinstance(profile_value, (int, float))
        and not isinstance(profile_value, bool)
        and math.isfinite(float(profile_value))
        else None
    )
    graph_estimate = est.usd_estimate_max
    result: dict[str, Any] = {
        "profile_planning_estimate_usd": profile_estimate,
        "graph_approval_estimate_usd": graph_estimate,
        "approval_basis": "materialized graph estimate",
        "blocks_approval": False,
    }
    if profile_estimate is None or graph_estimate is None:
        result.update(
            {
                "status": "not_comparable",
                "difference_usd": None,
                "tolerance_usd": None,
            }
        )
        return result
    tolerance = max(absolute_tolerance_usd, abs(graph_estimate) * relative_tolerance)
    difference = profile_estimate - graph_estimate
    result.update(
        {
            "status": (
                "matches_within_tolerance"
                if abs(difference) <= tolerance
                else "differs"
            ),
            "difference_usd": difference,
            "tolerance_usd": tolerance,
        }
    )
    return result


def approval_card(
    intent: RunIntent,
    report: LoweringReport,
    est: Estimate,
    freeze: ConstraintFreeze,
    *,
    resolved_seeds: Optional[int] = None,
    resolved_seeds_source: Optional[str] = None,
    policy: Optional[AccountPolicy] = None,
    estimate_reconciliation: Optional[Mapping[str, Any]] = None,
) -> str:
    """The card the user sees before anything spends.

    Every line tags where its value came from: STATED (the user said it),
    MEASURED, DERIVED, POLICY, PROFILE, or UNKNOWN. The card states N, the
    arms, the estimate, the ceiling, and the enforced approval scope. Seeds
    remain visible as intent data, but the graph estimate does not derive its
    workload from them.
    """
    account_policy = policy or default_account_policy()
    policy_cap = account_policy.cap_for(est.tier)
    c = intent.counts
    arms = ", ".join(intent.predictor_arms) if intent.predictor_arms else "UNRESOLVED: none stated"
    if intent.seeds_per_design is not None:
        seeds_line = f"{intent.seeds_per_design} per design (STATED)"
    elif resolved_seeds is not None:
        source = resolved_seeds_source or "profile template read this run"
        seeds_line = f"{resolved_seeds} per design (PROFILE TEMPLATE: {source})"
    else:
        seeds_line = "UNRESOLVED: none stated; graph estimate does not use this field"
    if intent.deadline is None:
        deadline_line = "none set"
    elif intent.deadline.duration_days is not None:
        deadline_line = f"{intent.deadline.duration_days} days (STATED; no config field carries it)"
    else:
        deadline_line = f"{intent.deadline.date_iso} (STATED; no config field carries it)"

    if est.usd_estimate_min is not None and est.usd_estimate_max is not None:
        money = (
            f"estimated total {est.usd_estimate_min:.2f} to {est.usd_estimate_max:.2f} USD "
            "at rates read this run"
        )
    elif est.gpu_seconds_min is None:
        money = "UNKNOWN: provider timing is unresolved for this tier"
    else:
        money = "UNKNOWN: no provider rate parameter this run, so dollars cannot be computed honestly"
    provider_lines = _approval_provider_lines(est)
    reconciliation_lines: list[str] = []
    if estimate_reconciliation is not None:
        profile_estimate = estimate_reconciliation.get("profile_planning_estimate_usd")
        graph_estimate = estimate_reconciliation.get("graph_approval_estimate_usd")
        if isinstance(profile_estimate, (int, float)):
            reconciliation_lines.append(
                f"profile estimate {float(profile_estimate):.2f} USD (PROFILE PLANNING ONLY)"
            )
        if estimate_reconciliation.get("status") == "differs" and isinstance(
            graph_estimate, (int, float)
        ):
            difference = estimate_reconciliation.get("difference_usd")
            reconciliation_lines.append(
                "reconciliation WARNING: profile differs from the materialized graph by "
                f"{abs(float(difference)):.2f} USD; approval uses {float(graph_estimate):.2f} USD"
            )

    lines = [
        "=" * 74,
        "APPROVAL REQUIRED FOR LOCAL EXECUTOR DISPATCH",
        "=" * 74,
        f"target         {intent.target.name}"
        + ("" if intent.target.dossier_path else "  [dossier NOT BOUND: cannot lower targets[0]]"),
        f"N delivered    {c.delivered_designs} designs (STATED)",
        f"               -> selection.final_count",
        f"backbones      "
        + (str(c.generated_backbones) if c.generated_backbones is not None else "UNSTATED")
        + "  [per-arm counts are profile-owned]",
        f"seqs/backbone  "
        + (str(c.sequences_per_backbone) if c.sequences_per_backbone is not None else "UNSTATED")
        + "  [profile-owned unless stated]",
        f"screened       "
        + (str(c.screened_candidates) if c.screened_candidates is not None else "UNRESOLVED")
        + "  [derived at run time from the novelty-filter artifact]",
        f"arms           {arms}",
        f"seeds          {seeds_line}"
        + ("; selected-profile replication rules are checked at composition" if intent.seeds_per_design else ""),
        f"binder length  "
        + (f"{intent.binder_length.minimum_length}-{intent.binder_length.maximum_length} residues (STATED)"
           if intent.binder_length else "UNRESOLVED: not stated; selected tool limits apply"),
        f"provider       {intent.provider_id or 'RESOLVED FROM MATERIALIZED PLAN'}",
        f"rounds         {intent.optimization_rounds if intent.optimization_rounds is not None else 'PROFILE-OWNED'}",
        f"objective      "
        + (
            f"{intent.objective_direction} {intent.objective_metric}"
            if intent.objective_metric is not None
            else "PROFILE-OWNED"
        ),
        "-" * 74,
        "ESTIMATE (an estimate, not a measurement)",
        "RULE 1         Every paid stage shows its resolved provider before approval, and all stage estimates count toward one dollar total.",
        f"{'provider time' if est.stage_count else 'gpu':<14}"
        + (
            (
                f"{est.gpu_seconds_min:.1f} to {est.gpu_seconds_max:.1f} billed seconds"
                if est.stage_count
                else f"{est.gpu_seconds_min:.1f} to {est.gpu_seconds_max:.1f} GPU seconds "
                f"({est.folds} per-call predictions; runner count UNKNOWN)"
            )
            if est.gpu_seconds_min is not None
            else "UNKNOWN: provider timing is unresolved"
        ),
        f"               {est.seconds_kind}",
        (
            f"stages         {len(est.covered_stage_ids)} bounded of {est.paid_stage_count} paid "
            f"and {est.stage_count} materialized"
            if est.stage_count
            else "stages         legacy scalar estimate; no materialized plan was supplied"
        ),
        *provider_lines,
        (
            "UNKNOWN STAGES " + ", ".join(est.unknown_stage_ids)
            if est.unknown_stage_ids
            else "UNKNOWN STAGES none"
        ),
        (
            "UPPER-BOUND WORKLOAD STAGES " + ", ".join(est.upper_bound_stage_ids)
            if est.upper_bound_stage_ids
            else "UPPER-BOUND WORKLOAD STAGES none"
        ),
        f"money          {money}",
        *reconciliation_lines,
        f"price source   {est.rate_source or 'UNKNOWN: no rate record supplied'}",
        f"               {SOURCE_PRICE_WARNING}",
        f"CAMPAIGN CAP   "
        + (f"{intent.spend_ceiling.amount_usd:.2f} USD (STATED)" if intent.spend_ceiling
           else "NONE STATED: approval refuses"),
        f"POLICY CAP     "
        + (
            f"{policy_cap.maximum_spend_usd:.2f} USD (POLICY) for tier {est.tier}, from "
            f"{policy_cap.source}"
            if policy_cap is not None
            else f"NONE: tier {est.tier} has no account policy cap"
        ),
        f"               {policy_cap.authorization_text()}" if policy_cap is not None
        else "               no account policy cap applies to this tier",
        f"EFFECTIVE CAP  "
        + (
            f"{effective_ceiling_usd(intent.spend_ceiling.amount_usd, est.tier, account_policy):.2f} "
            + (
                "USD (DERIVED): the lower of the campaign cap and the policy cap"
                if policy_cap is not None
                else "USD (DERIVED): the stated campaign cap; no account policy cap applies"
            )
            if intent.spend_ceiling is not None
            else "UNKNOWN: no campaign cap was stated"
        ),
        f"deadline       {deadline_line}",
        "-" * 74,
        "APPROVAL SCOPE",
        "1              the approval ledger stores the record, freeze projection, estimate and tier",
        "               for the materialized run",
        "2              the local executor checks that ledger before it starts a stage",
        "3              the executor checks cumulative recorded spend plus the next stage's",
        "               approved maximum before dispatch, then records accepted paid work",
        "-" * 74,
        "WHAT THIS OUTPUT IS",
        "claims         candidate-level computational predictions only. Nothing here demonstrates",
        "               binding, function, safety, or selectivity",
        "-" * 74,
        f"freeze         {freeze.digest}",
        "re-run note    same digest means the same constraints; a scientific change makes a new digest",
        "unfilled       see the lowering report for everything this intent cannot fill",
        "=" * 74,
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class ApprovalRecord:
    status: str
    approved_at: str
    approved_freeze_sha256: str
    intent_id: str


def approved_budget(
    intent: RunIntent,
    est: Estimate,
    policy: AccountPolicy,
    *,
    now_iso: str,
) -> dict[str, Any]:
    """Record both caps, the effective ceiling, and the policy that set it.

    A retry or a resume reuses the approval row rather than re-approving, so
    the authorization this block names is the same one for every attempt on the
    run. lane.verify_execution_approval matches a row on run_fingerprint and
    freeze_digest, and both are stable across attempts, so no second attempt can
    reach a second authorization. A reader who wants to check the authorization
    opens the record named by policy_record_path and compares its digest.

    The cumulative budget is enforced across Modal attempts.
    lane.enforce_spend_cap compares against running totals in the spend ledger,
    and the Modal path writes its own row: record_estimated_stage_spend runs the
    moment the provider accepts the job, alongside a second caller later in
    dispatch. So a Modal retry or resume re-checks the cap against a total that
    grew. One dispatched stage adds its cumulative total, and the next stage over
    the cap is refused before command launch and adds no ledger row.
    """
    campaign_ceiling = (
        intent.spend_ceiling.amount_usd if intent.spend_ceiling is not None else None
    )
    cap = policy.cap_for(est.tier, now_iso=now_iso)
    budget: dict[str, Any] = {
        "campaign_ceiling_usd": campaign_ceiling,
        "policy_cap_usd": cap.maximum_spend_usd if cap is not None else None,
        "policy_cap_source": cap.source if cap is not None else None,
        "policy_authorized_by": cap.authorized_by if cap is not None else None,
        "policy_provider_account": cap.provider_account if cap is not None else None,
        "policy_expires_at": cap.expires_at if cap is not None else None,
        "policy_record_path": policy.record_path,
        "policy_record_sha256": policy.record_sha256,
        "policy_read_at": policy.read_at,
        "tier": est.tier,
    }
    budget["effective_ceiling_usd"] = (
        None
        if campaign_ceiling is None
        else effective_ceiling_usd(campaign_ceiling, est.tier, policy, now_iso=now_iso)
    )
    return budget


def persist_approval(
    ledger_path: Path,
    record: ApprovalRecord,
    freeze: ConstraintFreeze,
    estimate: Estimate,
    *,
    run_fingerprint: str,
    tier: Optional[str] = None,
    budget: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Append one approval and its immutable inputs to the ledger."""
    if record.status != "approved":
        raise ValueError("only an approved record can enter the approval ledger")
    if record.approved_freeze_sha256 != freeze.digest:
        raise ValueError("approval record and freeze digest do not match")
    if not re.fullmatch(r"[0-9a-f]{64}", run_fingerprint):
        raise ValueError("run_fingerprint must be a lowercase SHA-256 digest")
    stored_tier = tier or estimate.tier
    if stored_tier != estimate.tier:
        raise ValueError("approval tier must match the estimate tier")
    projection = json.loads(json.dumps(freeze.projection))
    row: dict[str, Any] = {
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "freeze_digest": freeze.digest,
        "record": asdict(record),
        "projection": projection,
        "projection_sha256": _json_sha256(projection),
        "estimate": asdict(estimate),
        "tier": stored_tier,
    }
    if estimate.provider_records_required:
        row["paid_stage_providers"] = [
            json.loads(json.dumps(detail["provider_record"]))
            for detail in estimate.stage_estimates
            if detail.get("paid") is True and isinstance(detail.get("provider_record"), dict)
        ]
    if budget is not None:
        row["budget"] = json.loads(json.dumps(budget))
    row["row_sha256"] = _json_sha256(row)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_approval_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    """Load approval rows and refuse the ledger when any row was altered."""
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"approval ledger line {line_number} must be an object")
        row_hash = row.get("row_sha256")
        payload = dict(row)
        payload.pop("row_sha256", None)
        if not isinstance(row_hash, str) or _json_sha256(payload) != row_hash:
            raise ValueError(f"approval ledger line {line_number} failed its row hash")
        projection = row.get("projection")
        if not isinstance(projection, dict) or row.get("projection_sha256") != _json_sha256(projection):
            raise ValueError(f"approval ledger line {line_number} failed its projection hash")
        rows.append(row)
    return rows


def approval_timestamp(now: Optional[datetime] = None) -> str:
    """Return a UTC approval time rounded to whole seconds.

    The optional value lets a test supply a fixed instant. Production callers
    omit it, so the record captures the time when approval is requested.
    """
    instant = datetime.now(timezone.utc) if now is None else now
    if instant.tzinfo is None:
        raise ValueError("approval timestamp must include a UTC offset")
    utc_instant = instant.astimezone(timezone.utc).replace(microsecond=0)
    return utc_instant.isoformat().replace("+00:00", "Z")


def parse_approval_timestamp(value: str) -> str:
    """Validate and normalize a caller-supplied approval time to UTC seconds."""
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "--approved-at must be an ISO-8601 timestamp with a UTC offset"
        ) from exc
    return approval_timestamp(instant)


def resolve_planned_ledger(
    plan_path: Path,
    plan: dict[str, Any],
    requested_ledger: Path,
    freeze_digest: str,
) -> tuple[Path, str]:
    """Resolve the plan's ledger and bind it to a matching frozen plan.

    `--ledger` is an explicit write request. Its path must equal the relative
    approval ledger named by the materialized plan. This prevents a valid
    approval row from being written outside the bundle that the executor will
    later verify.
    """
    bundle_root = plan_path.resolve().parent
    ledger_name = plan.get("approval_ledger")
    if not isinstance(ledger_name, str) or not ledger_name:
        raise ValueError("the plan has no approval ledger path")
    ledger_relative = Path(ledger_name)
    if ledger_relative.is_absolute() or ".." in ledger_relative.parts:
        raise ValueError("the plan approval ledger path escapes the run bundle")
    planned_ledger = (bundle_root / ledger_relative).resolve()
    try:
        planned_ledger.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError("the plan approval ledger path escapes the run bundle") from exc
    if requested_ledger.resolve() != planned_ledger:
        raise ValueError(
            "--ledger must equal the approval ledger named by the plan: "
            f"{planned_ledger}"
        )
    run_fingerprint = plan.get("run_fingerprint")
    if not isinstance(run_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", run_fingerprint):
        raise ValueError("the plan has no valid run fingerprint")
    plan_freeze = plan.get("freeze_digest")
    if not isinstance(plan_freeze, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_freeze):
        raise ValueError("the plan has no valid freeze digest")
    if plan_freeze != freeze_digest:
        raise ValueError(
            "the plan freeze digest does not match the approval constraints; "
            "materialize the plan with this freeze digest before approving it"
        )
    return planned_ledger, run_fingerprint


def approve(
    intent: RunIntent,
    report: LoweringReport,
    est: Estimate,
    freeze: ConstraintFreeze,
    *,
    rate_available: bool,
    now_iso: str,
    ledger_path: Optional[Path] = None,
    run_fingerprint: Optional[str] = None,
    policy: Optional[AccountPolicy] = None,
) -> ApprovalRecord:
    """Gate approval. Returns a record only when every hard block clears.

    The campaign cap is mandatory and comes from the request. An installation
    may configure a second account-policy cap; when present, the lower of the
    two is the effective ceiling. With no account policy, the campaign cap binds.
    """
    blocks: list[str] = []
    account_policy = policy or default_account_policy()
    if est.designs != intent.counts.delivered_designs:
        blocks.append(
            f"estimate.designs={est.designs} differs from "
            f"requested.delivered_designs={intent.counts.delivered_designs}"
        )
    if est.provider_records_required:
        for detail in est.stage_estimates:
            if detail.get("paid") is not True or detail.get("provider_status") == "MATCH":
                continue
            stage_id = str(detail.get("stage_id", "<unnamed stage>"))
            recorded_ids = detail.get("recorded_provider_ids")
            resolved_ids = detail.get("provider_ids")
            recorded = _provider_display(
                recorded_ids if isinstance(recorded_ids, list) else []
            )
            resolved = _provider_display(
                resolved_ids if isinstance(resolved_ids, list) else []
            )
            blocks.append(
                f"paid stage {stage_id} provider approval is blocked: recorded provider "
                f"{recorded}, resolved provider {resolved}; {detail.get('provider_reason')}"
            )
    if est.unknown_stage_ids:
        unknown_details = []
        for detail in est.stage_estimates:
            if detail.get("status") != "UNKNOWN" or detail.get("paid") is not True:
                continue
            stage_id = str(detail.get("stage_id", "<unnamed stage>"))
            reason = str(detail.get("reason", "no bound was recorded"))
            required_value = str(detail.get("required_value", "a stage workload and timing upper bound"))
            unknown_details.append(f"{stage_id}: {reason}; provide {required_value}")
        blocks.append(
            "paid-stage bound unavailable: "
            + (" | ".join(unknown_details) if unknown_details else ", ".join(est.unknown_stage_ids))
        )
    if intent.spend_ceiling is None:
        blocks.append("no spend ceiling was stated")
    if not rate_available:
        blocks.append("price unavailable: no rate parameter, so the ceiling cannot be checked")
    elif est.tier == "modal" and est.gpu_seconds_max is None:
        blocks.append("timing unavailable: Modal estimate requires a measured Modal canary")
    elif est.usd_estimate_max is not None and intent.spend_ceiling is not None:
        ceiling = effective_ceiling_usd(
            intent.spend_ceiling.amount_usd,
            est.tier,
            account_policy,
            now_iso=now_iso,
        )
        if est.usd_estimate_max > ceiling:
            policy_cap = account_policy.cap_for(est.tier, now_iso=now_iso)
            if policy_cap is None:
                blocks.append(
                    f"maximum estimate {est.usd_estimate_max:.2f} USD exceeds the stated "
                    f"campaign ceiling {ceiling:.2f} USD"
                )
            else:
                blocks.append(
                    f"maximum estimate {est.usd_estimate_max:.2f} USD exceeds the effective "
                    f"ceiling {ceiling:.2f} USD, the lower of the campaign ceiling "
                    f"{intent.spend_ceiling.amount_usd:.2f} USD and the {est.tier} account "
                    f"policy cap"
                )
    if not intent.predictor_arms:
        blocks.append("predictor arms unresolved: name at least one arm")
    if intent.binder_length is None:
        blocks.append("binder length unresolved: state a positive ordered residue range")
    if intent.target.dossier_path is None:
        blocks.append("target dossier not bound: targets[0] cannot be lowered")
    if intent.seeds_per_design is not None and intent.seeds_per_design < MIN_CUSTOM_RESCORE_SEEDS:
        blocks.append(
            f"seeds_per_design={intent.seeds_per_design} is below the executor's "
            "nonempty seed requirement"
        )
    if blocks:
        # Fail closed because an unresolved approval input must remain a refusal
        # until the person supplies a value that the estimate can check.
        raise LoweringBlocked(blocks)
    record = ApprovalRecord(
        status="approved",
        approved_at=now_iso,
        approved_freeze_sha256=freeze.digest,
        intent_id=intent.intent_id,
    )
    if ledger_path is not None:
        if run_fingerprint is None:
            raise ValueError("run_fingerprint is required when persisting an approval")
        persist_approval(
            ledger_path,
            record,
            freeze,
            est,
            run_fingerprint=run_fingerprint,
            budget=approved_budget(intent, est, account_policy, now_iso=now_iso),
        )
    return record


def materialized_budget_error(intent: RunIntent, plan: dict[str, Any]) -> Optional[str]:
    """Refuse approval when the frozen plan does not carry the stated ceiling."""
    if intent.spend_ceiling is None:
        return None
    provider = plan.get("provider")
    provider_object = provider if isinstance(provider, dict) else {}
    budget = provider_object.get("budget")
    budget_object = budget if isinstance(budget, dict) else {}
    resolved_maximum = budget_object.get("maximum_spend_usd")
    stated_maximum = intent.spend_ceiling.amount_usd
    if (
        isinstance(resolved_maximum, (int, float))
        and not isinstance(resolved_maximum, bool)
        and float(resolved_maximum) == float(stated_maximum)
    ):
        return None
    if resolved_maximum is None:
        resolved_text = "missing"
    else:
        resolved_text = f"{resolved_maximum} USD"
    return (
        f"stated ceiling {stated_maximum:.2f} USD conflicts with materialized "
        "provider.budget.maximum_spend_usd "
        f"{resolved_text}"
    )


# ---------------------------------------------------------------------------
# Small demonstration CLI. Not a product surface; it shows the flow end to end.
# ---------------------------------------------------------------------------

_DEMO_SENTENCES = [
    # Underspecified on purpose: the parser must refuse and ask.
    "design binders for this target, a few designs, cheap please",
    # Complete: parse and lower, then explain how to continue to a materialized plan.
    (
        "design binders for IL2RA against this target, n equals 10, "
        "60-90 residues, use esmfold2-fast, 5 seeds each, "
        "under 200 dollars, within two days"
    ),
]


def _next_plan_command(sentence: str) -> str:
    """Render the concrete plan/rate command needed after an estimate refusal."""
    script = "run_intent.py"
    if sys.argv and str(sys.argv[0]).endswith("run_intent.py"):
        script = str(Path(sys.argv[0]).name)
    return " ".join(
        (
            shlex.quote(sys.executable),
            shlex.quote(script),
            "--say",
            shlex.quote(sentence),
            "--plan",
            "/path/to/run-bundle/run-plan.json",
            "--rate-record",
            "/path/to/planning-rate-record.json",
        )
    )


def _selftest() -> int:
    """Exercise refuse, parse, lower, freeze stability, and the refusal block."""
    failures: list[str] = []

    vague = parse_request(_DEMO_SENTENCES[0], context={"dossier_path": "targets/il2ra/dossier.json"})
    if vague.ok or not vague.questions:
        failures.append("vague request should refuse with questions")

    good = parse_request(
        _DEMO_SENTENCES[1],
        context={"dossier_path": "targets/il2ra/dossier.json"},
        known_arms=("esmfold2-fast",),
    )
    if not good.ok or good.intent is None:
        failures.append(f"complete request should parse, got questions: {good.questions}")
        print("\n\n".join(failures))
        return 1
    intent = good.intent
    checks = [
        intent.counts.delivered_designs == 10,
        intent.spend_ceiling is not None and intent.spend_ceiling.amount_usd == 200.0,
        intent.seeds_per_design == 5,
        intent.predictor_arms == ("esmfold2-fast",),
        intent.binder_length is not None and intent.binder_length.minimum_length == 60,
        intent.deadline is not None and intent.deadline.duration_days == 2,
    ]
    if not all(checks):
        failures.append(f"parse produced wrong values: {checks}")

    report = lower_intent(intent, profile=None)
    if report.overlay.get("selection", {}).get("final_count") != 10:
        failures.append("lowering must map delivered designs to selection.final_count")
    if not report.cannot_fill:
        failures.append("lowering must report unfilled fields honestly")

    freeze_one = freeze_constraints(intent, report)
    freeze_two = freeze_constraints(intent, report)
    if freeze_one.digest != freeze_two.digest:
        failures.append("freeze must be stable across identical calls")

    nudged = RunIntent(
        target=intent.target,
        counts=IntentCounts(delivered_designs=11),
        predictor_arms=intent.predictor_arms,
        seeds_per_design=intent.seeds_per_design,
        binder_length=intent.binder_length,
        spend_ceiling=intent.spend_ceiling,
        deadline=intent.deadline,
        intent_id=intent.intent_id,
    )
    if freeze_constraints(nudged, lower_intent(nudged)).digest == freeze_one.digest:
        failures.append("changing N must change the freeze digest")

    est_no_rate = estimate_run(10, 5, "fal")
    if est_no_rate.usd_estimate is not None or est_no_rate.rate_status != "UNKNOWN":
        failures.append("without a rate parameter dollars must stay UNKNOWN")
    expected_folds = 50
    expected_gpu_min = expected_folds * ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MIN
    expected_gpu_max = expected_folds * ESMFOLD2_FAST_BILLED_SECONDS_PER_FOLD_MAX
    if (
        abs(est_no_rate.gpu_seconds_min - expected_gpu_min) > 1e-9
        or abs(est_no_rate.gpu_seconds_max - expected_gpu_max) > 1e-9
        or est_no_rate.containers is not None
        or est_no_rate.shard_size is not None
    ):
        failures.append("fal estimate arithmetic mismatch")

    est_rate = estimate_run(10, 5, "fal", rate_usd_per_gpu_second=1000.0, rate_source="test-rate")
    if (
        est_rate.usd_estimate_min is None
        or est_rate.usd_estimate_max is None
        or est_rate.usd_estimate_min <= est_rate.gpu_seconds_min
    ):
        failures.append("rate parameter must convert gpu seconds to dollars")

    fixed_approval_time = datetime(2026, 8, 23, 12, 34, 56, 987654, tzinfo=timezone.utc)
    if approval_timestamp(fixed_approval_time) != "2026-08-23T12:34:56Z":
        failures.append("approval time must be UTC at second resolution")

    card = approval_card(intent, report, est_no_rate, freeze_one)
    for needed in ("10 designs", "esmfold2-fast", "UNKNOWN", "APPROVAL SCOPE"):
        if needed not in card:
            failures.append(f"approval card missing: {needed}")

    try:
        approve(intent, report, est_no_rate, freeze_one, rate_available=False, now_iso="now")
        failures.append("approval must refuse when no rate is available")
    except LoweringBlocked:
        pass

    print("selftest:", "PASS" if not failures else "FAIL")
    for problem in failures:
        print("  -", problem)
    return 0 if not failures else 1


def _campaign_context(
    config: Mapping[str, Any], source_path: Path
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Return target evidence and predictor IDs from one resolved campaign."""
    targets = config.get("targets")
    target_rows = [item for item in targets if isinstance(item, Mapping)] if isinstance(targets, list) else []
    target = next(
        (item for item in target_rows if item.get("role") == "primary"),
        target_rows[0] if target_rows else None,
    )
    context: dict[str, Any] = {}
    if isinstance(target, Mapping):
        target_name = target.get("target_id") or target.get("source_id")
        if isinstance(target_name, str) and target_name:
            context["target_name"] = target_name
        structure = target.get("structure_path") or target.get("structure_sha256")
        site = target.get("site")
        if (
            isinstance(target_name, str)
            and target_name
            and isinstance(structure, str)
            and structure
            and "__REQUIRED__" not in structure
            and isinstance(site, Mapping)
            and site
        ):
            # The resolved campaign is the evidence source for this target
            # record. It is real input, unlike the old hard-coded IL2RA path.
            context["dossier_path"] = str(source_path.resolve())

    cofold = config.get("cofold")
    predictors = cofold.get("predictors") if isinstance(cofold, Mapping) else None
    known_arms = tuple(
        str(item["id"])
        for item in (predictors if isinstance(predictors, list) else [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    )
    enabled_arms = tuple(
        str(item["id"])
        for item in (predictors if isinstance(predictors, list) else [])
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and item.get("enabled", True) is True
    )
    if enabled_arms:
        context["predictor_arms"] = enabled_arms
    binder = config.get("binder")
    if isinstance(binder, Mapping):
        minimum_length = binder.get("minimum_length")
        maximum_length = binder.get("maximum_length")
        if (
            isinstance(minimum_length, int)
            and not isinstance(minimum_length, bool)
            and isinstance(maximum_length, int)
            and not isinstance(maximum_length, bool)
        ):
            context["binder_length"] = (minimum_length, maximum_length)
    return context, known_arms


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--say", help='a plain-language request, e.g. "design binders, n=10, under 200 dollars"')
    parser.add_argument("--plan", type=Path, help="materialized run-plan.json used for the estimate")
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "resolved campaign configuration used for target, route, predictor, rounds, "
            "and objective choices; defaults to config.resolved.json beside --plan"
        ),
    )
    parser.add_argument(
        "--tier",
        choices=("modal", "fal", "runpod", "lambda"),
        help="provider tier recorded for this materialized plan",
    )
    parser.add_argument(
        "--rate-record",
        type=Path,
        help=(
            "JSON record with operator-attested per-machine planning rates, source, "
            "and read_at timestamp; it is not a settled billing record"
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="write approval only to the in-bundle ledger named by --plan",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        help="write the lowered config overlay for composition before materialization",
    )
    parser.add_argument(
        "--policy-record",
        type=Path,
        help=(
            "account policy record holding the account holder's per-tier spend caps; "
            f"defaults to ${ACCOUNT_POLICY_ENVIRONMENT_KEY}; if absent, no account cap applies"
        ),
    )
    parser.add_argument(
        "--approved-at",
        help="fixed ISO-8601 UTC approval time for a test; defaults to the current UTC time",
    )
    parser.add_argument("--demo", action="store_true", help="run the canned sentences end to end")
    parser.add_argument("--selftest", action="store_true", help="run the built-in checks")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    rate_record: Optional[RateRecord] = None
    if args.rate_record is not None:
        try:
            rate_record = load_rate_record(args.rate_record)
        except ValueError as exc:
            parser.error(str(exc))

    # A configured record that will not load is refused here rather than replaced
    # by the default, so a broken file never decides a cap.
    try:
        account_policy = resolve_account_policy(args.policy_record)
    except ValueError as exc:
        parser.error(str(exc))

    campaign_config: Optional[dict[str, Any]] = None
    campaign_config_path = args.config
    if campaign_config_path is None and args.plan is not None:
        sibling = args.plan.parent / "config.resolved.json"
        if sibling.is_file():
            campaign_config_path = sibling
    if campaign_config_path is not None:
        try:
            loaded_config = json.loads(campaign_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not read resolved campaign {campaign_config_path}: {exc}")
        if not isinstance(loaded_config, dict):
            parser.error(f"resolved campaign {campaign_config_path} must be a JSON object")
        campaign_config = loaded_config

    sentences = _DEMO_SENTENCES if args.demo else ([args.say] if args.say else [])
    if not sentences:
        parser.print_help()
        return 2

    if args.demo:
        context = {
            "dossier_path": "targets/il2ra/dossier.json",
            "target_name": "demo-il2ra",
        }
        configured_arms: Optional[tuple[str, ...]] = None
    elif campaign_config is not None and campaign_config_path is not None:
        context, discovered_arms = _campaign_context(campaign_config, campaign_config_path)
        configured_arms = discovered_arms
    else:
        context = {}
        configured_arms = None
    # A refusal that exits zero reads as success to anything checking a status code,
    # and the caller here is usually an agent rather than a person reading the text.
    refused = False
    for sentence in sentences:
        print(f'SAY: "{sentence}"')
        parsed = parse_request(
            sentence,
            context=context,
            known_arms=configured_arms,
            policy=account_policy,
        )
        if not parsed.ok:
            print("REFUSED: the request is incomplete. Ask:")
            for question in parsed.questions:
                print(f"  ? {question}")
            if parsed.understood:
                print("  captured so far: " + json.dumps(parsed.understood, sort_keys=True))
            print()
            refused = True
            continue
        intent = parsed.intent
        assert intent is not None
        try:
            report = lower_intent(intent, campaign_overrides=campaign_config)
        except LoweringBlocked as exc:
            print("LOWERED REFUSED:")
            for reason in exc.reasons:
                print(f"  ! {reason}")
            print()
            refused = True
            continue
        print("FILLED:")
        for key, target_field in report.filled.items():
            print(f"  {key} -> {target_field}")
        print("CANNOT FILL (asked, not guessed):")
        for target_field, reason in report.cannot_fill.items():
            print(f"  {target_field}: {reason}")
        if args.overlay is not None:
            try:
                write_intent_overlay(args.overlay, report)
            except OSError as exc:
                print(f"REFUSED: could not write intent overlay {args.overlay}: {exc}")
                refused = True
                print()
                continue
            print(f"INTENT OVERLAY: {args.overlay}")
        if args.plan is None:
            if args.overlay is not None:
                print(
                    "NO ESTIMATE: overlay written. Compose and materialize the plan, "
                    "then rerun with --plan for approval."
                )
                print("NEXT COMMAND: after materializing the plan and its rate record, run:")
                print(f"  {_next_plan_command(sentence)}")
                print()
                continue
            print(
                "NO ESTIMATE: a materialized execution plan is required. "
                "The campaign count is not a workload ceiling."
            )
            print("NEXT COMMAND: compose and materialize the lowered intent, then run:")
            print(f"  {_next_plan_command(sentence)}")
            print()
            refused = True
            continue
        try:
            plan = json.loads(args.plan.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"NO ESTIMATE: could not read materialized plan {args.plan}: {exc}")
            print()
            refused = True
            continue
        plan_tier = args.tier or plan.get("tier") or plan.get("provider_tier")
        if plan_tier not in {"fal", "modal", "runpod", "lambda"}:
            print("NO ESTIMATE: materialized plan has no provider tier; refusing to guess one")
            print()
            refused = True
            continue
        freeze = freeze_constraints(intent, report)
        try:
            est = estimate_from_graph(
                plan,
                str(plan_tier),
                rate_usd_per_gpu_second=(
                    rate_record.rates_usd_per_second.get(DEFAULT_SCALAR_RATE_MACHINE)
                    if rate_record is not None
                    else None
                ),
                rate_source=rate_record.source_label if rate_record is not None else None,
                rates_usd_per_machine_second=(
                    rate_record.rates_usd_per_second if rate_record is not None else None
                ),
                rate_machine=DEFAULT_SCALAR_RATE_MACHINE,
            )
        except ValueError as exc:
            refused = True
            print("APPROVAL BLOCKED:")
            print(f"  ! {exc}")
            print()
            continue
        try:
            validate_approval_design_counts(intent, plan, est)
        except (LoweringBlocked, ValueError) as exc:
            refused = True
            print("APPROVAL BLOCKED:")
            reasons = exc.reasons if isinstance(exc, LoweringBlocked) else [str(exc)]
            for reason in reasons:
                print(f"  ! {reason}")
            print()
            continue
        budget_error = materialized_budget_error(intent, plan)
        if budget_error is not None:
            refused = True
            print("APPROVAL BLOCKED:")
            print(f"  ! {budget_error}")
            print("  next      materialize the stated campaign cap, then rerun this command")
            print()
            continue
        estimate_reconciliation = materialized_estimate_reconciliation(plan, est)
        print(approval_card(
            intent, report, est, freeze,
            resolved_seeds=None,
            resolved_seeds_source=None,
            policy=account_policy,
            estimate_reconciliation=estimate_reconciliation,
        ))
        try:
            ledger_path: Optional[Path] = None
            run_fingerprint: Optional[str] = None
            if args.ledger is not None:
                ledger_path, run_fingerprint = resolve_planned_ledger(
                    args.plan, plan, args.ledger, freeze.digest
                )
            now_iso = (
                parse_approval_timestamp(args.approved_at)
                if args.approved_at is not None
                else approval_timestamp()
            )
            record = approve(
                intent,
                report,
                est,
                freeze,
                rate_available=rate_record is not None and est.usd_estimate_max is not None,
                now_iso=now_iso,
                ledger_path=ledger_path,
                run_fingerprint=run_fingerprint,
                policy=account_policy,
            )
            if ledger_path is None:
                print(f"APPROVED (NOT PERSISTED): {record.approved_freeze_sha256}")
            else:
                print(f"APPROVED AND WRITTEN: {record.approved_freeze_sha256}")
                print(f"APPROVAL LEDGER: {ledger_path}")
        except (LoweringBlocked, ValueError) as exc:
            refused = True
            print("APPROVAL BLOCKED:")
            reasons = exc.reasons if isinstance(exc, LoweringBlocked) else [str(exc)]
            for reason in reasons:
                print(f"  ! {reason}")
        print()
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
