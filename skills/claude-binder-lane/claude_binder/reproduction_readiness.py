#!/usr/bin/env python3
"""Report what a scientist must do next before a profile can dispatch.

``lane.PUBLISHED_BASELINE_TOOL_BINDINGS`` is the package's own statement of the
released campaign roster, and ``lane._baseline_fidelity`` withholds a baseline
claim until every one of those bindings is enabled under its published adapter
ID. This module derives, from shipped files alone, which of those bindings a
campaign could actually invoke today, and what stands between the profile and a
run.

Three fields answer three different questions, and none of them implies another.
``published_roster_bound`` says every roster entry has a bound adapter module.
``profile_configuration_complete`` says no value in the profile is still unset,
counting both the ``__REQUIRED__`` placeholders in adapter rows and the fal
deployment endpoints those rows reference through ``{{..._fal_url}}`` tokens.
``dispatchable_today`` is ``lane.compose_campaign`` run for real against the
shipped example campaign, which is the check that actually refuses a run. A
profile can be fully bound, wholly unconfigured and undispatchable at once, and
``rfdiffusion3-two-arm.template.json`` is exactly that profile. Each binding also
carries ``next_action``, the one thing to do about that row.

The dispatch probe costs local CPU and nothing else. It loads JSON, walks adapter
source with ``ast`` and writes one file into a temporary directory that is
removed. It opens no socket, starts no subprocess and calls no provider, so it
cannot bill anyone. It takes about nine seconds on a profile that composes and
about twenty milliseconds on one refused at the endpoint check, because a refusal
short-circuits before the stage-contract audit. ``--skip-dispatch-probe`` reports
``dispatchable_today: null`` instead.

**A withheld claim is not a blocked run.** ``baseline_fidelity: false`` stops no
stage and disables no tool. It prints one sentence in the report saying the run
does not claim fidelity, and it releases the campaign from four constraints that
only apply when fidelity is claimed: three generators, three predictor modes,
five rescore seeds and 50 scored candidates per generator, plus the forced
``ipsae_min`` and ``sc_dockq`` metrics. Both sites are in ``lane.validate_campaign``:
the ``if baseline_fidelity:`` branch that appends ``scoring.primary_metric must be
ipsae_min``, and the only error this flag raises, ``profile.baseline_fidelity=true
conflicts with the enabled generator, designer, or predictor bindings``, when a profile
declares a claim its own bindings contradict. Read it as a label on the claim, not as
a gate on execution.

Those two sites are named by their text rather than by a line number, because the
line number drifts. This docstring carried two line numbers that were already wrong,
and the sites they meant moved twice more inside one day of editing ``lane.py``. No
line number here names anything in another module, and ``test_reproduction_readiness``
checks both sites still sit in the function this paragraph names.

Every field here is read from the tree. Nothing is hand-maintained, because a
hand-maintained answer to this question has already gone stale once: a profile
recorded Genie3 as a configured method while its adapter row named no module.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import tempfile
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from . import lane
from .paths import package_root


MODULE_ARGV_RE = re.compile(r"^claude_binder\.adapters\.([a-z0-9_]+)$")
ARGV_KEYS = ("command_argv_template", "toolcheck_argv", "parser_argv_template")
DEFAULT_PROFILE = "full-ensemble.template.json"

#: The campaign the dispatch probe composes against. It is the one shipped campaign
#: `templates/README.md` hands a new user, so a refusal it produces is a refusal that
#: user would meet. A campaign of their own may resolve values this one leaves unset,
#: which is why the probe reports the campaign it used.
PROBE_CAMPAIGN = "campaign.example.json"

# A closed vocabulary. Each value states what the tree proves, and nothing more.
ADAPTER_SHIPPED = "adapter-shipped"
CONTRACT_ONLY = "contract-only"
UNBOUND_MODULE = "unbound-module"
ABSENT = "absent"

#: Older names for `published_roster_bound`, kept so a reader outside this package
#: does not break, and named here so they can be migrated. Both were documented as
#: meaning that every roster entry has a bound module, which is what the new name
#: says out loud.
DEPRECATED_FIELDS = {
    "baseline_reachable": "published_roster_bound",
    "binding_complete": "published_roster_bound",
}

ROLE_MODULE_SUFFIX = {
    "generators": "generator",
    "designers": "designer",
    "predictors": "predictor",
}


def profile_path(name: str = DEFAULT_PROFILE) -> Path:
    """Resolve a profile the way a scientist names it, not the way the file is spelled.

    `SKILL.md` move 3 names this module's `--profile` as the answer to decision 2, and
    a reader naming `rfdiffusion3-two-arm` there got a `FileNotFoundError` traceback
    rather than a report. The suffix is a packaging detail, so accept the bare stem,
    and refuse an unknown name by listing what is shipped instead of raising from the
    JSON loader three frames down.
    """
    directory = package_root() / "data" / "templates" / "profiles"
    for candidate in (name, f"{name}.template.json", f"{name}.json"):
        resolved = directory / candidate
        if resolved.is_file():
            return resolved
    shipped = sorted(path.name for path in directory.glob("*.json"))
    raise SystemExit(
        f"no shipped profile is named {name!r}. Available profiles:\n  "
        + "\n  ".join(shipped)
    )


def _module_name(adapter: dict[str, Any]) -> str | None:
    for key in ARGV_KEYS:
        for token in adapter.get(key) or []:
            match = MODULE_ARGV_RE.match(str(token))
            if match:
                return match.group(1)
    return None


def _placeholder_paths(node: Any) -> list[str]:
    """List every path inside one JSON value whose string is still unfilled.

    `lane.walk_strings` yields the path of every string at every depth, and
    `lane.is_required_placeholder` is the predicate the runtime already uses to
    refuse an unfilled value. Sharing both keeps the report and the run in
    agreement, which two shipped shapes broke while a narrower rule stood here.
    `environment_identity` carries the placeholder inside a longer string, as
    `modal-env:genie3_generator_gpu@spec_sha=__REQUIRED__`, so an equality test
    never saw it. `resources.container_image_digest` sits one level down, so a
    fixed tuple of top-level keys never reached it. Genie3 reported 0 against
    two real gaps, which tells a scientist a binding is ready that `materialize`
    refuses.

    Walking every depth is safe on the shipped tree. The placeholder appears at
    twelve distinct paths across the adapter rows of all 26 profiles, and each
    one is a value an operator supplies. No adapter row carries prose, and the
    profile-level `gap_notes` prose sits outside the adapters list.
    """
    return [
        path
        for path, value in lane.walk_strings(node)
        if lane.is_required_placeholder(value)
    ]


def _unresolved(adapter: dict[str, Any]) -> int:
    """Count the operator values one adapter row still needs before it can run."""
    return len(_placeholder_paths(adapter))


def _row_endpoint_fields(adapter: dict[str, Any]) -> list[str]:
    """Name the fal deployment endpoints one adapter row's own commands reference.

    This is the half of the question the placeholder walk cannot see. An argv
    element reading `{{rfdiffusion3_fal_url}}` carries no placeholder, so the row
    counted 0 unresolved values while `lane compose` refused the profile for the
    endpoint that token names. The token set comes from
    `lane._template_tokens`, the same function `lane._required_provider_endpoint_fields`
    uses, so a new endpoint field reaches this report without an edit here.
    """
    return sorted(set(lane.PROVIDER_ENDPOINT_FIELDS) & lane._template_tokens(adapter))


def _endpoint_refusals(profile: dict[str, Any]) -> dict[str, str]:
    """Map each unresolved endpoint field to the gate's own refusal sentence.

    `lane._provider_endpoint_errors` is what `compose_campaign` calls, so its text
    is the text a scientist will see when the run is refused. Reading it here rather
    than restating it is why the report and the gate cannot drift apart again.
    """
    refusals: dict[str, str] = {}
    for message in lane._provider_endpoint_errors(profile):
        for field in lane.PROVIDER_ENDPOINT_FIELDS.values():
            if message.startswith(f"provider_endpoints.{field} "):
                refusals.setdefault(field, message)
    return refusals


def _catalog_availability(catalog: dict[str, Any], tool_id: str) -> str | None:
    tool = (catalog.get("tools") or {}).get(tool_id)
    if not isinstance(tool, dict):
        return None
    availability = tool.get("availability")
    if not isinstance(availability, dict):
        return None
    status = availability.get("status")
    return status if isinstance(status, str) else None


def _assigned_tools(profile: dict[str, Any]) -> dict[str, str]:
    """Which tool each adapter id is actually filling in this profile.

    An adapter id is a role slot, so a profile can point `rfdiffusion-generator` at
    RFdiffusion3, and the small-run family does. Counting that row as the published
    RFdiffusion binding reported a profile as shipping a baseline tool it does not
    run, and made the shipped count exceed the selected one.
    """
    assigned: dict[str, str] = {}
    for section, key in (
        ("generation", "generators"),
        ("sequence_design", "designers"),
        ("cofold", "predictors"),
    ):
        block = profile.get(section)
        if not isinstance(block, dict):
            continue
        for item in block.get(key) or []:
            if (
                isinstance(item, dict)
                and isinstance(item.get("adapter_id"), str)
                and isinstance(item.get("id"), str)
            ):
                assigned[item["adapter_id"]] = item["id"]
    return assigned


@functools.lru_cache(maxsize=None)
def _profiles_binding(tool_id: str, adapter_id: str) -> tuple[str, ...]:
    """Name every shipped profile that does bind this published tool under this ID.

    A row this profile cannot reach is often one file away, and naming that file is
    the next action. The answer is derived by asking each shipped profile the same
    question this module asks, so it cannot disagree with the states it reports. It is
    cached per binding because the shipped tree does not change inside one process.
    """
    directory = package_root() / "data" / "templates" / "profiles"
    adapter_dir = package_root() / "adapters"
    naming: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            candidate = lane.load_profile(path)
        except Exception:
            # A contract-test fixture that no longer loads is not an answer to this
            # question, and refusing the whole report over one is worse than skipping it.
            continue
        rows = {
            row["adapter_id"]: row
            for row in candidate.get("adapters") or []
            if isinstance(row, dict) and isinstance(row.get("adapter_id"), str)
        }
        adapter = rows.get(adapter_id)
        if adapter is None:
            continue
        rebound_to = _assigned_tools(candidate).get(adapter_id)
        if rebound_to is not None and rebound_to != tool_id:
            continue
        module = _module_name(adapter)
        if module and (adapter_dir / f"{module}.py").is_file():
            naming.append(path.name)
    return tuple(naming)


def _elsewhere(tool_id: str, adapter_id: str, this_profile: str) -> str:
    """Say which other shipped profile binds this row, when one does."""
    others = [name for name in _profiles_binding(tool_id, adapter_id) if name != this_profile]
    if not others:
        return " No shipped profile binds it."
    return f" Or run --profile {others[0]}, which binds it."


# The roster this package ships records what a real run reported for three
# adapters. It is not an authorization: the roster's own qualification status is
# `UNVERIFIABLE_TARGET_NOT_SHIPPED`, because the structure bytes those records
# were measured against are absent from the package. The values are still
# observations from a deployment that ran, which is what makes them worth
# offering, and the sentence below says which of the two it is. Those are exactly the fields a profile leaves at
# `__REQUIRED__`, so a reader who has to fill one can be handed the measured
# value instead of a research task. It is offered and never assumed: a scientist
# deploying something other than the qualified application has a different
# identity, and only they know which.
_ROSTER_FIELD_SOURCES = {
    "source_revision": ("source_revision",),
    "environment_identity": ("environment_identity",),
    "model_revision": ("weights_revision", "weights_sha256"),
}


@functools.lru_cache(maxsize=1)
def _roster_qualification_status() -> str:
    """Return the roster's own qualification status, so the offer can name it."""
    try:
        roster = json.loads(
            (lane.package_file("data", "model-roster.json")).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return "unknown"
    status = (roster.get("qualification") or {}).get("status")
    return status if isinstance(status, str) and status else "unknown"


@functools.lru_cache(maxsize=1)
def _qualified_roster_values() -> dict[str, dict[str, str]]:
    """Return {adapter_id: {template field: measured value}} from the shipped roster."""
    try:
        roster = json.loads(
            (lane.package_file("data", "model-roster.json")).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    found: dict[str, dict[str, str]] = {}
    for entry in roster.get("models", []):
        if not isinstance(entry, Mapping):
            continue
        adapter_id = entry.get("adapter_id")
        if not isinstance(adapter_id, str):
            continue
        fields: dict[str, str] = {}
        for template_field, roster_keys in _ROSTER_FIELD_SOURCES.items():
            for key in roster_keys:
                value = entry.get(key)
                if isinstance(value, str) and value and not value.startswith(
                    "not_applicable"
                ):
                    fields[template_field] = value
                    break
        if fields:
            found[adapter_id] = fields
    return found


def _measured_value_hint(adapter_id: str, field_paths: Sequence[str]) -> str:
    """Return the sentence offering every measured value this row still needs.

    It reads every unresolved path in the row rather than only the one the action
    leads with, because the leading path is chosen for how well `lane` can
    describe it and has nothing to do with which values were measured.
    """
    measured = _qualified_roster_values().get(adapter_id, {})
    if not measured:
        return ""
    offered: list[str] = []
    for path in field_paths:
        field = path.split(".")[0].split("[")[0]
        value = measured.get(field)
        if not value or any(field == name for name, _ in (item.split(" ", 1) for item in offered)):
            continue
        shortened = value if len(value) <= 110 else value[:107] + "..."
        offered.append(f"{field} `{shortened}`")
    if not offered:
        return ""
    listed = "; ".join(offered)
    subject = "this value" if len(offered) == 1 else "these values"
    return (
        f" A recorded run of this adapter reported {subject}, in "
        f"`data/model-roster.json`: {listed}. That roster's own qualification status is "
        f"`{_roster_qualification_status()}`, so these are observations from that "
        f"deployment and not an authorization to run. Use them when your deployment is "
        f"that one, and record your own when it is not."
    )


def _next_action(
    row: dict[str, Any],
    adapter: dict[str, Any] | None,
    profile: dict[str, Any],
    adapter_index: int | None,
    endpoint_refusals: dict[str, str],
    profile_file: str,
) -> str:
    """The one thing to do about this binding, derived from the row itself.

    The order follows the order the runtime refuses in. `compose_campaign` checks
    provider endpoints before it validates the merged campaign, so an unset endpoint
    is what a scientist meets first and is named first here.
    """
    tool_id, adapter_id = row["tool_id"], row["adapter_id"]
    state = row["binding_state"]
    if state == ABSENT:
        return (
            f"Nothing in this package binds {tool_id}: this profile carries no "
            f"`{adapter_id}` row and no `{row['expected_module']}.py` ships."
        )
    if state == UNBOUND_MODULE:
        return (
            f"`{row['unbound_module']}.py` ships and this profile does not bind it. "
            f"Add the `{adapter_id}` row whose command names "
            f"`claude_binder.adapters.{row['unbound_module']}`."
            + _elsewhere(tool_id, adapter_id, profile_file)
        )
    if state == CONTRACT_ONLY:
        return (
            f"The `{adapter_id}` row names no `claude_binder.adapters` module. Point "
            "its command_argv_template at one."
            + _elsewhere(tool_id, adapter_id, profile_file)
        )
    unresolved_endpoints = row["unresolved_provider_endpoints"]
    if unresolved_endpoints:
        first = endpoint_refusals.get(unresolved_endpoints[0])
        if first:
            # The gate's own sentence, verbatim. It already reads as an instruction,
            # and a scientist who then runs `lane compose` sees the same words back.
            return f"{first}."
    paths = row["unresolved_operator_value_paths"]
    if paths:
        # Prefer an argv element whose option `lane` can name. Every path in the list
        # has to be filled, so which one leads is a choice, and the one the runtime can
        # describe in its own words is worth more to a reader than a bare JSON key.
        chosen, context = paths[0], ""
        if adapter_index is not None:
            for path in paths:
                candidate = lane._argv_placeholder_context(
                    profile, f"adapters[{adapter_index}].{path}"
                )
                if candidate:
                    chosen, context = path, candidate
                    break
        if context:
            # `lane` composes this sentence for its own refusal, naming the option the
            # element supplies and the adapter --help that documents the value.
            action = f"Fill `{chosen}` in the `{adapter_id}` row{context}"
        else:
            current = dict(lane.walk_strings(adapter or {})).get(chosen)
            shown = f" It reads `{current}` today." if isinstance(current, str) else ""
            action = f"Fill `{chosen}` in the `{adapter_id}` row.{shown}"
        action += _measured_value_hint(adapter_id, paths)
        remaining = len(paths) - 1
        if remaining:
            tail = "value remains" if remaining == 1 else "values remain"
            action += f" {remaining} more unset {tail} in this row."
        return action
    return f"Nothing is unset in the `{adapter_id}` row."


def binding_rows(profile: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Join the published roster to the profile, the adapter modules, and the catalog.

    `next_action` is not set here. `lane._baseline_fidelity_details` calls this function
    while it validates a campaign and reads `binding_state` alone, and deriving an action
    reads every shipped profile. `report` attaches the actions afterwards, so the
    validation path pays nothing for a string it does not use.
    """
    adapter_list = [
        row
        for row in profile.get("adapters") or []
        if isinstance(row, dict) and isinstance(row.get("adapter_id"), str)
    ]
    adapters = {row["adapter_id"]: row for row in adapter_list}
    adapter_index = {
        row["adapter_id"]: index
        for index, row in enumerate(profile.get("adapters") or [])
        if isinstance(row, dict) and isinstance(row.get("adapter_id"), str)
    }
    adapter_dir = package_root() / "adapters"
    assigned_tool = _assigned_tools(profile)
    endpoint_refusals = _endpoint_refusals(profile)
    rows: list[dict[str, Any]] = []
    for role, bindings in lane.PUBLISHED_BASELINE_TOOL_BINDINGS.items():
        for tool_id, adapter_id in bindings:
            adapter = adapters.get(adapter_id)
            # A profile that assigns this adapter to another tool does not satisfy this
            # published binding, whatever module the adapter names. A profile that makes
            # no assignment is unchanged: the row still reports from adapter presence.
            rebound_to = assigned_tool.get(adapter_id)
            if rebound_to is not None and rebound_to != tool_id:
                adapter = None
            module = _module_name(adapter) if adapter else None
            module_present = bool(module) and (adapter_dir / f"{module}.py").is_file()
            suffix = ROLE_MODULE_SUFFIX[role]
            candidate = f"{tool_id.replace('-', '_')}_{suffix}"
            unbound = (adapter_dir / f"{candidate}.py").is_file()
            if module_present:
                state = ADAPTER_SHIPPED
            elif adapter is not None:
                state = CONTRACT_ONLY
            elif unbound:
                # The module ships, but no profile row binds it under the
                # published adapter ID, so a baseline run cannot reach it.
                state = UNBOUND_MODULE
            # `unbound_module` below is set from the same file test rather than
            # from the state, because a contract-only row can also sit beside a
            # shipped module it does not name. Genie3 was that shape until
            # 2026-09-12, when full-ensemble's argv was filled and the row became
            # adapter-shipped. A profile that removes the adapter still leaves
            # `genie3_generator.py` in this directory, so the module has to be
            # reported separately from the state.
            else:
                state = ABSENT
            qualification = (adapter or {}).get("qualification") or {}
            cost_basis = qualification.get("cost_basis") or {}
            endpoint_fields = _row_endpoint_fields(adapter) if adapter else []
            placeholder_paths = _placeholder_paths(adapter) if adapter else []
            row = {
                "role": role,
                "tool_id": tool_id,
                "adapter_id": adapter_id,
                "binding_state": state,
                "profile_row": adapter is not None,
                "adapter_module": module,
                "adapter_module_present": module_present,
                "expected_module": candidate,
                "unbound_module": candidate if unbound and candidate != module else None,
                "unresolved_operator_values": len(placeholder_paths) if adapter else None,
                "unresolved_operator_value_paths": placeholder_paths,
                "required_provider_endpoints": endpoint_fields,
                "unresolved_provider_endpoints": [
                    field for field in endpoint_fields if field in endpoint_refusals
                ],
                "cost_basis_kind": cost_basis.get("kind"),
                "catalog_availability": _catalog_availability(catalog, tool_id),
            }
            rows.append(row)
    return rows


def attach_next_actions(
    rows: list[dict[str, Any]], profile: dict[str, Any], profile_file: str = ""
) -> list[dict[str, Any]]:
    """Give every binding the one thing to do about it.

    `profile_file` is the shipped file name the profile was read from. It is used only
    to keep an action from offering the profile the reader already selected.
    """
    adapters = {
        row["adapter_id"]: row
        for row in profile.get("adapters") or []
        if isinstance(row, dict) and isinstance(row.get("adapter_id"), str)
    }
    adapter_index = {
        row["adapter_id"]: index
        for index, row in enumerate(profile.get("adapters") or [])
        if isinstance(row, dict) and isinstance(row.get("adapter_id"), str)
    }
    endpoint_refusals = _endpoint_refusals(profile)
    for row in rows:
        adapter = adapters.get(row["adapter_id"]) if row["profile_row"] else None
        row["next_action"] = _next_action(
            row,
            adapter,
            profile,
            adapter_index.get(row["adapter_id"]) if adapter is not None else None,
            endpoint_refusals,
            profile_file,
        )
    return rows


@functools.lru_cache(maxsize=None)
def _dispatch_probe(profile_file: str) -> tuple[bool, tuple[str, ...]]:
    """Compose this profile against the shipped example campaign and report the verdict.

    This is the honest check. The report used to answer from adapter rows alone and
    said a profile was ready while `lane compose` refused it by name, because the
    gate reads five top-level `provider_endpoints` fields the row walk never saw.
    Running the gate removes the possibility of that disagreement rather than
    patching the one case that produced it.

    The composed plan is written into a temporary directory and discarded. Nothing
    outside that directory is touched, and the probe makes no network call, so it is
    safe to run on every invocation. The result is cached per profile file because
    the tree does not change inside one process.
    """
    campaign = package_root() / "data" / "templates" / PROBE_CAMPAIGN
    with tempfile.TemporaryDirectory(prefix="binder-readiness-") as directory:
        result = lane.compose_campaign(
            campaign, Path(profile_file), Path(directory) / "composed.json"
        )
    errors = result.get("errors") or []
    return bool(result.get("ok")), tuple(str(error) for error in errors)


def report(
    profile_name: str = DEFAULT_PROFILE, *, dispatch_probe: bool = True
) -> dict[str, Any]:
    path = profile_path(profile_name)
    # Read through the same loader the executor uses. A profile that states
    # base_profile carries its adapters in an overlay, and reading the file
    # literally found no adapters at all, so every binding in an overlay profile
    # reported as unreachable. That is the hand-maintained answer this module
    # exists to replace, arriving by a different route.
    profile = lane.load_profile(path)
    catalog = json.loads((package_root() / "data" / "catalog.json").read_text(encoding="utf-8"))
    rows = attach_next_actions(binding_rows(profile, catalog), profile, path.name)
    shipped = [row for row in rows if row["binding_state"] == ADAPTER_SHIPPED]
    blocking = [row for row in rows if row["binding_state"] != ADAPTER_SHIPPED]
    priced = [row for row in shipped if row["cost_basis_kind"] == "settled billed amount"]
    roster_bound = not blocking

    # Two scopes, both stated, because they answer different questions. The roster
    # count is what decision 2 quotes. The profile-wide count is what a run needs,
    # and it is larger, because a profile carries adapter rows outside the published
    # twelve and those rows run too.
    in_bindings = sum(row["unresolved_operator_values"] or 0 for row in rows)
    in_profile = len(_placeholder_paths(profile.get("adapters") or []))
    endpoint_refusals = _endpoint_refusals(profile)
    unresolved_endpoints = sorted(endpoint_refusals)
    configuration_complete = in_profile == 0 and not unresolved_endpoints

    if dispatch_probe:
        dispatchable, dispatch_errors = _dispatch_probe(str(path))
        probe = {
            "status": "assessed",
            "campaign": PROBE_CAMPAIGN,
            "gate": "lane.compose_campaign",
            "blockers": list(dispatch_errors),
        }
    else:
        dispatchable, probe = None, {
            "status": "not_assessed",
            "campaign": PROBE_CAMPAIGN,
            "gate": "lane.compose_campaign",
            "blockers": [],
        }

    return {
        "schema_version": 2,
        "check_type": "claude_binder_reproduction_readiness",
        "assessment_scope": (
            "Shipped module bindings, unset profile values, and whether this profile "
            f"composes against {PROBE_CAMPAIGN}. A completed provider run, scientific "
            "qualification and exact protocol reproduction are not established by this report."
        ),
        "profile": profile_name,
        "profile_baseline_fidelity": (profile.get("profile") or {}).get("baseline_fidelity"),
        "published_bindings": len(rows),
        "adapter_shipped": len(shipped),
        "blocking_bindings": [row["adapter_id"] for row in blocking],
        "settled_price_bindings": [row["adapter_id"] for row in priced],
        # The three headline answers. Each names its own question, and none implies
        # another. A single boolean stood here and said `baseline_reachable: true` for
        # a profile `lane compose` refuses by name.
        "published_roster_bound": roster_bound,
        "profile_configuration_complete": configuration_complete,
        "dispatchable_today": dispatchable,
        "outstanding_configuration": {
            "operator_values_in_published_bindings": in_bindings,
            "operator_values_in_all_adapter_rows": in_profile,
            "unresolved_provider_endpoints": unresolved_endpoints,
            "blocking_published_bindings": in_bindings + len(unresolved_endpoints),
        },
        "dispatch_probe": probe,
        "execution_verification": "not_assessed",
        # Compatibility. `references/reproduction-readiness.md` documents both as
        # meaning that every roster entry has a bound module, which is what
        # `published_roster_bound` now says in its name. Removing them would break a
        # reader outside this package silently, so they stay and point at the new name.
        "baseline_reachable": roster_bound,
        "binding_complete": roster_bound,
        "deprecated_fields": dict(DEPRECATED_FIELDS),
        "bindings": rows,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Role | Tool | Adapter ID | Binding state | Module | Operator values | Endpoints | Cost basis |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["bindings"]:
        unresolved = row["unresolved_operator_values"]
        endpoints = row["unresolved_provider_endpoints"]
        lines.append(
            "| {role} | {tool_id} | `{adapter_id}` | {binding_state} | {module} | {unresolved} "
            "| {endpoints} | {cost} |".format(
                module=(
                    f"`{row['adapter_module']}`"
                    if row["adapter_module"]
                    else (f"`{row['unbound_module']}` (unbound)" if row["unbound_module"] else "none")
                ),
                unresolved="-" if unresolved is None else unresolved,
                endpoints=", ".join(f"`{field}`" for field in endpoints) or "-",
                cost=row["cost_basis_kind"] or "-",
                **row,
            )
        )
    outstanding = result["outstanding_configuration"]
    dispatchable = result["dispatchable_today"]
    lines += [
        "",
        f"- **published_roster_bound** {str(result['published_roster_bound']).lower()}",
        f"- **profile_configuration_complete** {str(result['profile_configuration_complete']).lower()}",
        "- **dispatchable_today** "
        + ("not assessed" if dispatchable is None else str(dispatchable).lower())
        + f" ({result['dispatch_probe']['gate']} against {result['dispatch_probe']['campaign']})",
        f"- **outstanding values** {outstanding['blocking_published_bindings']} across the "
        f"published bindings: {outstanding['operator_values_in_published_bindings']} operator "
        f"values and {len(outstanding['unresolved_provider_endpoints'])} deployment endpoints",
        "",
        "## Next action per binding",
        "",
    ]
    for row in result["bindings"]:
        lines.append(f"- `{row['adapter_id']}` {row['next_action']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="profile template file name")
    parser.add_argument("--markdown", action="store_true", help="print the binding table")
    parser.add_argument(
        "--skip-dispatch-probe",
        action="store_true",
        help="do not compose the profile; dispatchable_today is reported as null",
    )
    args = parser.parse_args(argv)
    result = report(args.profile, dispatch_probe=not args.skip_dispatch_probe)
    if args.markdown:
        print(markdown(result))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
