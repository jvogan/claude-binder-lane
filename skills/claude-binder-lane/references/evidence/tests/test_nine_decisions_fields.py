"""Keep configuration paths in the nine-decisions reference code-owned."""

from __future__ import annotations

import copy
import inspect
import json
import re
from pathlib import Path

from claude_binder import lane, ranking_policy
from claude_binder.paths import package_root

# The build ships this test beside `skill_layout.py` under
# `references/evidence/tests/`, where it is a loose file rather than part of the
# `claude_binder.tests` package, so the package import is tried first and the
# sibling import is what an installed skill uses.
try:
    from claude_binder.tests.skill_layout import skill_root
except ImportError:  # pragma: no cover - taken only in an installed skill
    from skill_layout import skill_root


CONFIGURATION_ROOTS = {
    "cofold",
    "filters",
    "generation",
    "optimization",
    "profile",
    "provider",
    "runtime",
    "scoring",
    "selection",
    "sequence_design",
    "targets",
    "viewer_renderer",
}
FIELD_PATH = re.compile(
    r"(?P<quote>[`'])(?P<field>[a-z][a-z0-9_]*(?:\[\])?"
    r"(?:\.[a-z][a-z0-9_]*(?:\[\])?)+|viewer_renderer)(?P=quote)"
)


def documented_configuration_fields(document: str) -> set[str]:
    """Return configuration paths named by the reference document."""
    fields = {match.group("field") for match in FIELD_PATH.finditer(document)}
    return {
        field
        for field in fields
        if field.split(".", 1)[0].removesuffix("[]") in CONFIGURATION_ROOTS
    }


def schema_path_exists(schema: object, path: str) -> bool:
    """Return whether a representative resolved configuration owns a path."""
    value = schema
    for raw_segment in path.split("."):
        is_list = raw_segment.endswith("[]")
        segment = raw_segment.removesuffix("[]")
        if not isinstance(value, dict) or segment not in value:
            return False
        value = value[segment]
        if is_list:
            if not isinstance(value, list) or not value:
                return False
            value = value[0]
    return True


def resolved_campaign_schema(tmp_path: Path) -> dict[str, object]:
    """Compose the shipped local contract into one validated schema example."""
    templates = package_root() / "data" / "templates"
    composed_path = tmp_path / "composed.json"
    result = lane.compose_campaign(
        templates / "fixtures" / "local-contract" / "campaign.json",
        templates / "profiles" / "local-contract-test.json",
        composed_path,
    )
    assert result["ok"], result
    configuration = json.loads(composed_path.read_text(encoding="utf-8"))
    configuration["filters"].setdefault("disabled_checks", [])
    configuration["viewer_renderer"] = "browser"
    configuration["provider"] = {
        "provider_id": "local",
        "surface": "local fixture",
        "workspace": "local fixture",
        "credential_lane": "none",
        "artifact_return_mode": "local files",
        "access": "private",
        "persistent_root": "{{data_root}}",
        "budget": {
            "estimated_spend_usd": 1,
            "maximum_spend_usd": 1,
            "currency": "USD",
            "pricing_source": "fixture",
            "pricing_checked_at": "fixture",
            "ceiling_stated_by": "fixture",
        },
        "lifecycle": {
            "no_retry": True,
            "verify_bundle_hash_before_run": True,
            "verify_artifact_hashes_after_run": True,
            "request_timeout_seconds": 0,
            "keep_alive_seconds": 0,
        },
    }
    check = lane.validate_campaign(configuration, config_path=composed_path)
    assert check["ok"], check

    schema = copy.deepcopy(configuration)
    schema["profile"]["baseline_fidelity"] = False
    # The field-map audit walks optional union branches too. Populate their
    # representative paths after validating the real fixture, so this setup
    # does not pretend that one site uses every mutually exclusive source.
    site = schema["targets"][0]["site"]
    site.update(
        {
            "resolution_artifact_path": "site-resolution.json",
            "hotspot_source": "explicit",
            "hotspot_residues": ["A:1"],
            "published_epitope_table": "epitope.tsv",
            "published_epitope_chain_policy": "as-written",
        }
    )
    schema["optimization"]["early_stop_margin_metric"] = "ipsae_min"
    # A campaign may gate its own objective by registering the metric and
    # gating it. Both keys are optional and the local fixture sets neither,
    # so populate representative values the way the site branches above do.
    schema["scoring"]["metric_registry"] = [
        {
            "metric_id": "buried-surface-area",
            "reducer": "mean",
            "minimum": 0.0,
            "maximum": 1.0,
        }
    ]
    schema["scoring"]["custom_metric_gates"] = [
        {
            "metric": "buried-surface-area",
            "operator": "minimum",
            "threshold": 0.5,
        }
    ]
    schema["scoring"]["ranking_mode"] = ranking_policy.CUSTOM_WEIGHTED_RANKING_MODE
    schema["selection"]["diversity"] = lane.default_diversity_policy(schema)
    schema["targets"][0]["site"]["discovery"] = {
        "max_designs": 1,
        "max_folds": 1,
    }
    return schema


def test_every_documented_configuration_field_exists_in_campaign_schema(tmp_path: Path) -> None:
    document = (skill_root() / "references" / "the-nine-decisions.md").read_text(
        encoding="utf-8"
    )
    documented_fields = documented_configuration_fields(document)
    assert documented_fields

    schema = resolved_campaign_schema(tmp_path)
    missing = sorted(
        field for field in documented_fields if not schema_path_exists(schema, field)
    )
    assert missing == []

    validator_source = "\n".join(
        (
            inspect.getsource(lane.validate_campaign),
            inspect.getsource(lane.compose_campaign),
            inspect.getsource(lane.estimate_fanout),
            inspect.getsource(lane.selection_reachability),
            inspect.getsource(ranking_policy._custom_mode_reason),
        )
    )
    unowned = sorted(
        field
        for field in documented_fields
        if field != "viewer_renderer" and field.rsplit(".", 1)[1].removesuffix("[]") not in validator_source
    )
    assert unowned == []
