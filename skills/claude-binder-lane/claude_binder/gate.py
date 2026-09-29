#!/usr/bin/env python3
"""Standalone licence gate for claude_binder campaign configurations.

This module imports nothing from any repository. It needs Python 3.10+ and the
standard library only, so it can be reviewed, copied, or unit-tested on its own.

Purpose
-------
The profiles let a scientist select pipeline tools by editing JSON by hand
(``generation.generators``, ``sequence_design.designers``, ``cofold.predictors``).
Nothing in the runtime checks what those selections are allowed to do. This
module closes that gap: given a composed campaign/profile configuration and a
tool catalog (see ``catalog.json`` beside this file), it returns the licence
problems for the declared use.

It handles at least these cases, on the code licence and on the model weights
separately:

1. A tool whose licence forbids the declared use.
2. A tool whose licence is unknown or unsettled (fail closed for commercial use).
3. A tool that needs a separate written agreement before commercial use.
   Worked case: UCSF ChimeraX is distributed under the UCSF ChimeraX
   Non-Commercial License Agreement; commercial use requires a separately
   executed written licence with UCSF. The gate refuses such a campaign unless
   the configuration records an agreement reference, and then downgrades to a
   verify-scope warning.
4. A tool whose licence permits the use under conditions. It clears only when
   the catalog states each condition, and the pass carries them as a warning.
5. A value outside the vocabulary the catalog declares at ``commercial_use_enum``.
   The check is an allowlist, so a misspelled or invented licence value refuses
   rather than clearing. It was a denylist of unsettled values on the weights
   side until 2026-09-12, which cleared ``forbidden``, an absent field, and any
   unrecognized word.
6. A weight set whose licence permits the use and whose own provenance status
   says the weights reading is not finished. That is a separate finding from an
   unresolved licence, so it carries the separate code
   ``WEIGHTS_PROVENANCE_UNVERIFIED``. The status is checked against the
   vocabulary the catalog declares at ``weights_status_enum``, so a null, an
   absent key, or an unrecognized spelling refuses under
   ``WEIGHTS_STATUS_UNRECOGNIZED``. This check was a denylist of four spellings
   until 2026-09-12, which cleared ``unverified`` and every invented word.
7. A code licence whose own status says the licence reading is not finished,
   under ``LICENCE_PROVENANCE_UNVERIFIED``, and a status outside the vocabulary
   the catalog declares at ``code_licence_status_enum``, under
   ``LICENCE_STATUS_UNRECOGNIZED``. ``code_licence.status`` was read by nothing
   until 2026-09-12, so ``unverified``, ``open_todo``, a null, an absent key and
   any invented word all cleared.

Where the vocabularies come from
--------------------------------
The catalog declares each vocabulary and this module implements each value's
meaning, so the gate accepts a value only when both say so. A catalog that
declares fewer values narrows what the gate accepts. A catalog that declares
more does not widen it: a declared value this module implements no rule for is
not in the vocabulary the gate checks against, so it refuses under the axis's
``_UNRECOGNIZED`` code rather than falling through to a pass. That matters
because ``catalog.json`` is the file this gate polices. Until 2026-09-12 a
one-line edit adding ``perrmited`` to ``commercial_use_enum`` cleared a
commercial campaign that used it, and the same edit against
``weights_status_enum`` cleared a weights row whose provenance nobody had read.

Semantics
---------
* ``declared_use`` comes from the configuration key ``declared_use``
  ("commercial" | "non-commercial") or the ``declared_use`` argument.
* An unset or unknown declared use produces a ``DECLARED_USE_REQUIRED`` error.
  The check also evaluates commercial risk conservatively so a caller cannot
  mistake an undeclared campaign for a permitted non-commercial campaign. Those
  conservative findings are reported as warnings whose code carries an
  ``_IF_COMMERCIAL`` suffix, because they describe what one answer would cost
  rather than a fault in the configuration. The campaign is still refused: the
  undeclared use is the error, and the preview tells the scientist what each
  answer does to the selected tool set before they choose.
* Any error makes ``report["ok"]`` false. Callers should refuse to materialize
  or execute a campaign whose gate report is not ok.
* Every problem carries the catalog citation that produced it, so a refusal can
  be traced to opened evidence rather than to this module's opinion.

Worked example
--------------

    >>> report = evaluate(
    ...     {"declared_use": "commercial",
    ...      "generation": {"generators": []},
    ...      "sequence_design": {"designers": []},
    ...      "cofold": {"predictors": []}},
    ...     CATALOG_FIXTURE,
    ... )
    >>> [p["code"] for p in report["problems"]]
    ['WRITTEN_AGREEMENT_REQUIRED']

With ``licences.chimerax.agreement_reference`` recorded the same campaign
passes with a warning instead of an error. Run ``python3 gate.py --self-test``
for the sixteen built-in scenarios, including this one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from claude_binder.refusals import ExitCode, Refusal, exit_code_for_result

__version__ = "0.1.0"

DECLARED_USE_KEY = "declared_use"
AGREEMENT_SECTION_KEY = "licences"
AGREEMENT_FIELD = "agreement_reference"

# Selection sections of a composed configuration that name tools directly.
SELECTION_SECTIONS = (
    ("generation", "generators"),
    ("sequence_design", "designers"),
    ("cofold", "predictors"),
)

# Tools implied unconditionally by an enabled adapter, beyond the direct
# selection sections. Sourced from the shipped profile capabilities. The MSA
# builder and viewer renderer are absent because their tool sets depend on
# command arguments that collect_enabled_tools reads separately.
ADAPTER_TO_TOOLS = {
    "interface-scorer": ("ipsae", "dockq"),
    "novelty-filter": ("mmseqs2", "uniref90"),
}

_MSA_SOURCE_TOKEN = "--source"
_MSA_ROUTE_TOOLS = {
    "public-server": ("colabfold-msa-server",),
    "local": ("mmseqs2",),
}

# The catalog declares its own commercial-use vocabulary at
# catalog.json:"commercial_use_enum". The gate reads that declaration so a value
# outside it refuses instead of clearing. A licence field is the wrong place for
# a permissive default: a typo there would otherwise read as permission.
COMMERCIAL_USE_ENUM_KEY = "commercial_use_enum"
# The four commercial-use values this module implements a rule for: "permitted"
# clears, "permitted_with_conditions" clears only with the conditions stated,
# "forbidden" refuses, "unknown" refuses. A catalog that declares a fifth value
# does not get it accepted, because _vocabulary intersects the declaration with
# this tuple. Until 2026-09-12 it did not, so adding "perrmited" to
# commercial_use_enum and writing "perrmited" in a tool's licence row cleared a
# commercial campaign. Extend this tuple only together with the branch in
# evaluate that decides what the new value does.
GATE_COMMERCIAL_USE_VALUES = (
    "permitted",
    "permitted_with_conditions",
    "forbidden",
    "unknown",
)
# catalog.json's purpose line declares __REQUIRED__ for a field no opened file
# settles. MISSING is the second spelling of the same hole, which catalog.json's
# own notes record two weights rows as having carried. Both are holes rather
# than licence positions, so the declared vocabulary omits them and the gate
# treats them as unsettled. This module has recognized both since v0.1.0.
PLACEHOLDER_VALUES = frozenset({"__REQUIRED__", "MISSING"})
# A weights layer that does not exist. Catalog rows carry it in
# weights, and references/licence-and-commercial-use.md defines it as "the named
# tool has no model-weight layer in the 2026 evidence". It answers no licence
# question for code, so only the weights check accepts it.
NOT_APPLICABLE = "N/A"
# A value that names a hole, plus the absent field itself.
UNSETTLED_COMMERCIAL_VALUES = frozenset({None, "unknown"}) | PLACEHOLDER_VALUES
# The catalog declares its own code-licence status vocabulary at
# catalog.json:"code_licence_status_enum", and the gate reads it the way it reads
# the other two. The declaration was derived from the distinct values in the
# catalogue. Enumerating those values settles no licence question.
CODE_LICENCE_STATUS_ENUM_KEY = "code_licence_status_enum"
# The six code-licence statuses this module implements a rule for: the four
# spellings the catalog's rows use, plus the two placeholder spellings above.
# catalog.json's purpose line declares __REQUIRED__ for any field no opened file
# settles, and MISSING is the second spelling of the same hole, which pxdesign
# carries today at code_licence.spdx. A status records how far the licence
# reading got, and "nobody recorded it" is one answer that reading can reach, so
# both placeholders belong in this vocabulary.
#
# Lower-case "missing" is deliberately absent. It is in the weights vocabulary
# because the pre-2026-09-12 weights denylist had named it since v0.1.0. No
# code_licence row uses it and no code here ever read it, so adding it would be
# inventing a spelling. A row that reaches it refuses as an unrecognized status,
# which is the fail-closed answer.
GATE_CODE_LICENCE_STATUS_VALUES = (
    "resolved",
    "resolved_licence_unknown_commercial",
    "unverified",
    "open_todo",
    "__REQUIRED__",
    "MISSING",
)
# The statuses that leave the code-licence record unread. "unverified" and
# "open_todo" say so in the word, and the two placeholder spellings name a field
# nobody filled in. An open record is not a prohibition, so a row carrying one
# refuses under LICENCE_PROVENANCE_UNVERIFIED and never under a code that reads
# as a ban.
#
# "resolved" stays outside this set. So does resolved_licence_unknown_commercial,
# which is the verdict this module gives every status it does not call open:
#
# TODO(provenance): whether resolved_licence_unknown_commercial leaves the code
# licence open. rfdiffusion3 is the one row carrying it. Its code_licence records
# BSD-3-Clause with commercial_use "unknown", and its own evidence string states
# that no file in this repository states the RFdiffusion3 code licence, so the
# spdx value and the evidence disagree. Its "unknown" commercial_use already
# refuses every commercial campaign that selects it, so the classification moves
# no verdict today. Settled by an RFdiffusion3 row in references/tool-licences.md
# read from the rc-foundry LICENSE.md at the pinned revision
# rc-foundry 0.2.0+app-09dd49945f0e6142, which is the read that row's evidence
# names.
#
# There is no N/A guard here, unlike the weights side. Every catalogued tool has
# code, so "the layer does not exist" is not an answer a code licence can give,
# and no shipped code_licence row records N/A.
OPEN_CODE_LICENCE_RECORD = frozenset({"unverified", "open_todo"}) | PLACEHOLDER_VALUES
# The catalog declares its own weights-provenance vocabulary at
# catalog.json:"weights_status_enum", and the gate reads it the same way it reads
# the commercial-use one. The declaration was derived from the distinct values
# in the catalogue. Enumerating those values settles no licence question.
WEIGHTS_STATUS_ENUM_KEY = "weights_status_enum"
# The ten weights statuses this module implements a rule for: the seven spellings
# catalog.json's rows use, plus the three placeholder spellings below. A status
# records how far the weights reading got, and "nobody recorded it" is one of the
# answers that reading can reach, so the placeholders belong in this vocabulary
# where they do not belong in the commercial-use one. A catalog that declares an
# eleventh value does not get it accepted, because _vocabulary intersects the
# declaration with this tuple. Extend the tuple only together with a decision,
# recorded below, about whether the new value leaves provenance open.
GATE_WEIGHTS_STATUS_VALUES = (
    "resolved",
    "resolved_with_condition",
    "resolved_with_open_question",
    "resolved_licence_unknown_commercial",
    "not_applicable",
    "unverified",
    "open_todo",
    "missing",
    "__REQUIRED__",
    "MISSING",
)
# The statuses that leave the weight artifact unsettled. "unverified" and
# "open_todo" say so in the word, and the three placeholder spellings name a
# field nobody filled in. This set was a denylist until 2026-09-12, so every
# other spelling cleared the check, "unverified" and a typo included.
#
# The four resolved_* spellings stay outside this set, which is the verdict this
# module has always given them. Two of them carry a question the tree does not
# answer:
#
# TODO(provenance): whether resolved_with_open_question leaves the artifact open.
# Its two rows record different open questions. catalog.json's boltz row records
# that nobody has opened model-gateway.boltz.bio for the terms document served
# with the weights, and its boltz-local row records that the adapter requires an
# operator-recorded SHA-256 and refuses the stage on mismatch. Settled by that
# gateway terms document, read and recorded as a Boltz weights row in
# references/tool-licences.md.
#
# TODO(provenance): whether resolved_with_condition asserts that the artifact is
# identified. Its alphafold-multimer-v3 row states the condition "the deployed
# app must record an immutable artifact revision or SHA-256 for the five selected
# weights" in weights.note, and gate.conditions carries only the CC-BY-4.0
# attribution, so the gate hands the scientist one of the two. Settled by the
# AlphaFold2 model-parameters terms recorded as a condition at
# catalog.tools.alphafold-multimer-v3.gate.conditions, which is the field this
# module reads.
#
# TODO(provenance): whether resolved_licence_unknown_commercial asserts anything
# about the artifact. esmfold2-native-design is the one row carrying it, it names
# a Hugging Face host with no revision and no digest, and its weights
# commercial_use of "unknown" already refuses every commercial campaign, so the
# classification moves no verdict today. Settled by a design-specific licence
# source for the Biohub ESMFold2 weights, which line 89 of
# skills/claude-binder-lane/references/licence-and-commercial-use.md records as
# absent, on a row that reads unknown at high confidence.
OPEN_WEIGHT_PROVENANCE = (
    frozenset({"missing", "open_todo", "unverified"}) | PLACEHOLDER_VALUES
)


class GateError(RuntimeError):
    """The inputs cannot be evaluated."""


def _vocabulary(
    catalog: dict[str, Any], key: str, implemented: tuple[str, ...]
) -> frozenset[str]:
    """Return the declared values this module also implements a rule for.

    The catalog is the document this gate polices, so its declaration narrows the
    vocabulary and never widens it. A value the catalog declares and this module
    has no branch for is dropped here, which makes it unrecognized at the check
    and refuses the campaign. Until 2026-09-12 the declaration was taken whole,
    so adding one word to an enum in catalog.json cleared a campaign that used
    that word.

    A catalog that declares no usable vocabulary, or whose declaration names none
    of the implemented values, is evaluated against all of them. Every value is
    then still one this module decides, so that path cannot clear a campaign the
    implemented set refuses.
    """
    declared = catalog.get(key)
    if not isinstance(declared, list):
        return frozenset(implemented)
    values = {item for item in declared if isinstance(item, str) and item.strip()}
    return frozenset(values & set(implemented)) or frozenset(implemented)


def commercial_use_vocabulary(catalog: dict[str, Any]) -> frozenset[str]:
    """Return the commercial-use values the catalog declares and the gate implements."""
    return _vocabulary(catalog, COMMERCIAL_USE_ENUM_KEY, GATE_COMMERCIAL_USE_VALUES)


def code_licence_status_vocabulary(catalog: dict[str, Any]) -> frozenset[str]:
    """Return the code-licence statuses the catalog declares and the gate implements."""
    return _vocabulary(
        catalog, CODE_LICENCE_STATUS_ENUM_KEY, GATE_CODE_LICENCE_STATUS_VALUES
    )


def weights_status_vocabulary(catalog: dict[str, Any]) -> frozenset[str]:
    """Return the weights statuses the catalog declares and the gate implements."""
    return _vocabulary(catalog, WEIGHTS_STATUS_ENUM_KEY, GATE_WEIGHTS_STATUS_VALUES)


def stated_conditions(entry: dict[str, Any]) -> list[str]:
    """Return the conditions one catalog entry states, in recorded order.

    ``permitted_with_conditions`` is only usable when the conditions are
    recorded, which is the rule references/licence-and-commercial-use.md states
    for that value. This is the field that carries them.
    """
    gate_block = entry.get("gate")
    if not isinstance(gate_block, dict):
        return []
    conditions = gate_block.get("conditions")
    if not isinstance(conditions, list):
        return []
    return [item.strip() for item in conditions if isinstance(item, str) and item.strip()]


def load_catalog(path: str | Path) -> dict[str, Any]:
    """Load a catalog document from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("tools"), dict):
        raise GateError(f"{path}: catalog must be an object with a 'tools' object")
    return catalog


def extend_catalog(catalog: dict[str, Any], extensions: Any) -> dict[str, Any]:
    """Add campaign-local tool rows without changing or overriding shipped rows.

    The rows travel inside the campaign configuration and its run identity. A
    new model still needs a configured adapter and the normal provider approval.
    """
    if extensions is None:
        return catalog
    if not isinstance(extensions, dict):
        raise GateError("tool_catalog_extensions must be an object keyed by tool id")
    if not extensions:
        return catalog
    tools = catalog["tools"]
    added: dict[str, dict[str, Any]] = {}
    for tool_id, row in extensions.items():
        if not isinstance(tool_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", tool_id):
            raise GateError(f"tool_catalog_extensions: invalid tool id {tool_id!r}")
        if tool_id in tools:
            raise GateError(f"tool_catalog_extensions.{tool_id}: cannot replace a packaged tool")
        if not isinstance(row, dict):
            raise GateError(f"tool_catalog_extensions.{tool_id} must be an object")
        for field in ("display_name", "stage_category"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise GateError(f"tool_catalog_extensions.{tool_id}.{field} is required")
        for field, status_values in (
            ("code_licence", code_licence_status_vocabulary(catalog)),
            ("weights", weights_status_vocabulary(catalog)),
        ):
            layer = row.get(field)
            prefix = f"tool_catalog_extensions.{tool_id}.{field}"
            if not isinstance(layer, dict):
                raise GateError(f"{prefix} must be an object")
            status = layer.get("status")
            if not isinstance(status, str) or status not in status_values:
                raise GateError(f"{prefix}.status must use a recognised catalogue value")
            expected_use = NOT_APPLICABLE if field == "weights" and status == "not_applicable" else None
            if expected_use is not None:
                if layer.get("commercial_use") != expected_use:
                    raise GateError(f"{prefix}.commercial_use must be {expected_use!r}")
            elif (
                not isinstance(layer.get("commercial_use"), str)
                or layer["commercial_use"] not in commercial_use_vocabulary(catalog)
            ):
                raise GateError(f"{prefix}.commercial_use must use a recognised catalogue value")
            evidence = layer.get("evidence")
            if not isinstance(evidence, list) or not evidence or any(
                not isinstance(item, str) or not item.strip() for item in evidence
            ):
                raise GateError(f"{prefix}.evidence must contain a source citation")
        gate = row.get("gate")
        if not isinstance(gate, dict):
            raise GateError(f"tool_catalog_extensions.{tool_id}.gate must be an object")
        if not isinstance(gate.get("requires_written_agreement_for_commercial"), bool):
            raise GateError(
                f"tool_catalog_extensions.{tool_id}.gate.requires_written_agreement_for_commercial must be boolean"
            )
        conditions = gate.get("conditions")
        if not isinstance(conditions, list) or any(
            not isinstance(item, str) or not item.strip() for item in conditions
        ):
            raise GateError(f"tool_catalog_extensions.{tool_id}.gate.conditions must be a string list")
        added[tool_id] = row
    return {**catalog, "tools": {**tools, **added}}


def _enabled_items(section: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if isinstance(section, dict):
        # Accept either the section object itself ({"designers": [...]}) or a
        # parent that already holds one of the selection lists.
        for key in ("generators", "designers", "predictors"):
            value = section.get(key)
            if isinstance(value, list):
                items.extend(v for v in value if isinstance(v, dict))
    elif isinstance(section, list):
        items.extend(v for v in section if isinstance(v, dict))
    return [item for item in items if item.get("enabled", True)]


def collect_enabled_tools(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the tools a composed configuration selects, directly or by adapter.

    Each record carries how the tool was selected so a refusal can point at the
    exact JSON key a scientist would edit.
    """
    found: dict[str, dict[str, Any]] = {}
    profile = config.get("profile")
    synthetic_contract_tools_allowed = (
        isinstance(profile, dict) and profile.get("claim_level") == "contract-test"
    )
    for parent_key, child_key in SELECTION_SECTIONS:
        parent = config.get(parent_key)
        child = parent.get(child_key) if isinstance(parent, dict) else None
        for item in _enabled_items(child):
            # A supplied candidate is input data, represented in generation so the
            # scale and lineage contracts can count it. It invokes no catalog tool.
            if parent_key == "generation" and item.get("source") == "supplied":
                continue
            tool_id = item.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                continue
            if synthetic_contract_tools_allowed and tool_id.startswith("fixture-"):
                continue
            record = {
                "tool_id": tool_id,
                "selected_via": f"{parent_key}.{child_key}[id={tool_id}]",
                "adapter_id": item.get("adapter_id"),
            }
            found.setdefault(tool_id, record)

    adapters = config.get("adapters")
    adapter_ids: list[str] = []
    if isinstance(adapters, list):
        for adapter in adapters:
            if isinstance(adapter, dict) and isinstance(adapter.get("adapter_id"), str):
                adapter_ids.append(adapter["adapter_id"])
                role = adapter.get("role")
                if isinstance(role, str):
                    adapter_ids.append(role)

    for adapter_id in adapter_ids:
        for tool_id in ADAPTER_TO_TOOLS.get(adapter_id, ()):
            found.setdefault(
                tool_id,
                {
                    "tool_id": tool_id,
                    "selected_via": f"adapters[adapter_id={adapter_id}]",
                    "adapter_id": adapter_id,
                },
            )

    # The MSA builder's third-party exposure depends on its own --source token.
    for adapter_id, command in _iter_adapter_commands(config, "msa-builder"):
        for tool_id in _collect_msa_route_tools(command):
            found.setdefault(
                tool_id,
                {
                    "tool_id": tool_id,
                    "selected_via": f"adapters[{adapter_id}] --source route",
                    "adapter_id": adapter_id,
                },
            )

    # The shipped renderer accepts a PyMOL executable through --pymol. ChimeraX
    # is a cataloged option for separate viewer scripts, but this adapter does
    # not invoke it.
    for adapter_id, command in _iter_adapter_commands(config, "viewer-renderer"):
        if _command_option_value(command, "--pymol") == "pymol":
            found.setdefault(
                "pymol-open-source",
                {
                    "tool_id": "pymol-open-source",
                    "selected_via": f"adapters[{adapter_id}] --pymol renderer",
                    "adapter_id": adapter_id,
                },
            )

    return sorted(found.values(), key=lambda record: record["tool_id"])


def _iter_adapter_commands(config: dict[str, Any], adapter_id: str):
    adapters = config.get("adapters")
    if not isinstance(adapters, list):
        return
    for index, adapter in enumerate(adapters):
        if not isinstance(adapter, dict):
            continue
        if adapter_id not in (adapter.get("adapter_id"), adapter.get("role")):
            continue
        command = adapter.get("command_argv_template")
        yield f"adapters[{index}].command_argv_template", command


def _collect_msa_route_tools(command: Any) -> tuple[str, ...]:
    """Map an msa-builder --source token to the tools that route touches."""
    if not isinstance(command, list):
        return ()
    for index, token in enumerate(command[:-1]):
        if token == _MSA_SOURCE_TOKEN:
            route = command[index + 1]
            return _MSA_ROUTE_TOOLS.get(route, ())
    return ()


def _command_option_value(command: Any, option: str) -> str | None:
    if not isinstance(command, list):
        return None
    for index, token in enumerate(command[:-1]):
        if token == option and isinstance(command[index + 1], str):
            return command[index + 1]
    return None


def resolve_declared_use(
    config: dict[str, Any], override: str | None = None
) -> str:
    """Resolve the declared use; unknown values stay unknown."""
    value = override or config.get(DECLARED_USE_KEY)
    if isinstance(value, str) and value.lower() in {"commercial", "non-commercial"}:
        return value.lower()
    return "unknown"


def _agreement_reference(config: dict[str, Any], tool_id: str) -> str | None:
    section = config.get(AGREEMENT_SECTION_KEY)
    if not isinstance(section, dict):
        return None
    entry = section.get(tool_id)
    if isinstance(entry, dict):
        value = entry.get(AGREEMENT_FIELD)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _citation(entry: dict[str, Any], *keys: str) -> str:
    """First evidence line from the deepest matching catalog field."""
    node: Any = entry
    for key in keys:
        if not isinstance(node, dict):
            break
        node = node.get(key)
    if isinstance(node, dict):
        evidence = node.get("evidence")
        if isinstance(evidence, list) and evidence:
            return str(evidence[0])
    return "catalog entry carries no citation"


def _problem(
    severity: str,
    code: str,
    tool_id: str | None,
    message: str,
    *,
    source: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "code": code,
        "tool_id": tool_id or "",
        "message": message,
        "source": source,
    }


def _refusal_for_problem(problem: dict[str, str]) -> Refusal:
    """Return the operator-facing refusal for one licence finding."""
    code = problem["code"]
    tool_id = problem["tool_id"]
    tool_text = f" for tool {tool_id}" if tool_id else ""
    expected_by_code = {
        "DECLARED_USE_REQUIRED": "The configuration must declare commercial or non-commercial use.",
        "TOOL_NOT_IN_CATALOG": "The selected tool must have a catalog entry with evidence.",
        "WRITTEN_AGREEMENT_REQUIRED": "Commercial use must carry the required written agreement reference.",
        "CATALOG_GATE_CONFLICT": "The catalog commercial-use value and written-agreement gate must agree.",
        "COMMERCIAL_USE_FORBIDDEN": "The selected tool licence must permit the declared use.",
        "LICENCE_UNKNOWN_COMMERCIAL_USE": "The catalog must cite commercial-use terms for the selected tool.",
        "COMMERCIAL_USE_UNRECOGNIZED": (
            "The catalog commercial_use value must be one that both the catalog's "
            "own commercial_use_enum declares and this gate implements a rule for."
        ),
        "LICENCE_STATUS_UNRECOGNIZED": (
            "The code licence's status value must be one that both the catalog's "
            "own code_licence_status_enum declares and this gate implements a rule for."
        ),
        "LICENCE_PROVENANCE_UNVERIFIED": (
            "The catalog must settle the licence record for a value that already "
            "permits the use: which licence text was read, and any reading still "
            "open on it."
        ),
        "LICENCE_CONDITION_UNSTATED": (
            "A tool whose code licence is permitted_with_conditions must state "
            "each condition the scientist has to satisfy."
        ),
        "WEIGHTS_COMMERCIAL_USE_FORBIDDEN": (
            "The selected model weights' licence must permit the declared use."
        ),
        "WEIGHTS_COMMERCIAL_USE_UNKNOWN": (
            "The catalog must cite a commercial-use term for the selected model weights."
        ),
        "WEIGHTS_PROVENANCE_UNVERIFIED": (
            "The catalog must settle the weights record for a licence that already "
            "permits the use: which artifact runs, and any reading still open on it."
        ),
        "WEIGHTS_STATUS_UNRECOGNIZED": (
            "The model weights' status value must be one that both the catalog's "
            "own weights_status_enum declares and this gate implements a rule for."
        ),
        "WEIGHTS_COMMERCIAL_USE_UNRECOGNIZED": (
            "The model weights' commercial_use value must be one that both the "
            "catalog's own commercial_use_enum declares and this gate implements a "
            "rule for, or N/A where no weights exist."
        ),
        "WEIGHTS_CONDITION_UNSTATED": (
            "Model weights recorded as permitted_with_conditions must state each "
            "condition the scientist has to satisfy."
        ),
    }
    action_by_code = {
        "DECLARED_USE_REQUIRED": 'Set declared_use to "commercial" or "non-commercial".',
        "TOOL_NOT_IN_CATALOG": "Add an evidenced catalog entry for the selected tool.",
        "WRITTEN_AGREEMENT_REQUIRED": "Record the executed agreement reference in licences.",
        "CATALOG_GATE_CONFLICT": (
            "Correct the catalog commercial_use value or the gate requirement from the "
            "cited evidence before evaluating this campaign."
        ),
        "COMMERCIAL_USE_FORBIDDEN": "Select a tool whose licence permits the declared use.",
        "WEIGHTS_COMMERCIAL_USE_FORBIDDEN": (
            "Remove the tool from the selection named above. No agreement reference "
            "lifts a weights licence that forbids commercial use."
        ),
        "COMMERCIAL_USE_UNRECOGNIZED": (
            "Correct the catalog commercial_use value to one of the recognized "
            "values from the cited evidence, or remove the tool."
        ),
        "LICENCE_STATUS_UNRECOGNIZED": (
            "Correct the code licence status to one of the recognized values from "
            "the cited evidence, or remove the tool."
        ),
        "LICENCE_PROVENANCE_UNVERIFIED": (
            "Cite the licence text in the catalog entry, or close the reading its "
            "status names, or remove the tool from the selection named above."
        ),
        "WEIGHTS_COMMERCIAL_USE_UNRECOGNIZED": (
            "Correct the weights commercial_use value to one of the declared "
            "values from the cited evidence, or remove the tool."
        ),
        "LICENCE_CONDITION_UNSTATED": (
            "Record each condition in the catalog entry's gate.conditions from the "
            "cited licence, or remove the tool."
        ),
        "WEIGHTS_CONDITION_UNSTATED": (
            "Record each weights condition in the catalog entry's gate.conditions "
            "from the cited licence, or remove the tool."
        ),
        # A scientist reading this refusal can remove the tool now. Editing the
        # catalog needs evidence they may not hold, so it is the second option.
        "LICENCE_UNKNOWN_COMMERCIAL_USE": (
            "Remove the tool from the selection named above, or add a cited "
            "commercial-use term to its catalog entry."
        ),
        "WEIGHTS_COMMERCIAL_USE_UNKNOWN": (
            "Remove the tool from the selection named above, or add a cited "
            "commercial-use term to its catalog weights entry."
        ),
        "WEIGHTS_PROVENANCE_UNVERIFIED": (
            "Record the weight artifact's revision or SHA-256 in the catalog entry, "
            "or close the reading its status names, or remove the tool from the "
            "selection named above."
        ),
        "WEIGHTS_STATUS_UNRECOGNIZED": (
            "Correct the weights status to one of the recognized values from the cited "
            "evidence, or remove the tool."
        ),
    }
    return Refusal(
        cause=f"The licence gate rejected {code}{tool_text}.",
        expected=expected_by_code.get(code, "The licence gate must receive a supported configuration."),
        expected_source=problem["source"],
        found=problem["message"],
        found_source=problem["source"],
        scope="The licence gate evaluated the configuration and started no provider command.",
        action=action_by_code.get(code, "Correct the reported licence gate finding."),
        escalation="Send the configuration or catalog correction to the campaign maintainer.",
    )


def evaluate(
    config: dict[str, Any],
    catalog: dict[str, Any],
    *,
    declared_use: str | None = None,
) -> dict[str, Any]:
    """Evaluate the licence gates for one composed configuration.

    Returns a report dict; ``report["ok"]`` is false when any error-severity
    problem exists. Never raises for a normal bad configuration: problems are
    data, not exceptions.
    """
    tools_section = catalog.get("tools")
    if not isinstance(tools_section, dict):
        raise GateError("catalog must contain a 'tools' object")

    use = resolve_declared_use(config, declared_use)
    commercial_risk = use != "non-commercial"
    vocabulary = commercial_use_vocabulary(catalog)
    code_status_vocabulary = code_licence_status_vocabulary(catalog)
    status_vocabulary = weights_status_vocabulary(catalog)

    problems: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    # Findings that exist only because the use is or may be commercial. An
    # undeclared campaign still produces them, because the evaluation below stays
    # conservative, but they answer "what would commercial cost me" rather than
    # "what is wrong with this configuration". Routing them separately keeps the
    # one decision the scientist has to make from arriving under a stack of
    # findings that a single word makes go away.
    commercial_problems: list[dict[str, str]] = []
    commercial_warnings: list[dict[str, str]] = []

    selected = collect_enabled_tools(config)
    checked: list[dict[str, str]] = []

    for selection in selected:
        tool_id = selection["tool_id"]
        checked.append({"tool_id": tool_id, "selected_via": selection["selected_via"]})
        entry = tools_section.get(tool_id)
        if entry is None:
            problems.append(
                _problem(
                    "error",
                    "TOOL_NOT_IN_CATALOG",
                    tool_id,
                    f"tool '{tool_id}' selected via {selection['selected_via']} is not in "
                    "the catalog; add a fully evidenced row before enabling it",
                    source=f"configuration.{selection['selected_via']} and catalog.tools",
                )
            )
            continue

        if not commercial_risk:
            continue

        code_licence = entry.get("code_licence", {})
        weights = entry.get("weights", {})
        gate = entry.get("gate", {})

        needs_agreement = bool(gate.get("requires_written_agreement_for_commercial"))
        commercial_value = code_licence.get("commercial_use")
        code_status = code_licence.get("status")
        conditions = stated_conditions(entry)

        # The code-licence status is checked against the declared vocabulary on
        # its own, because an unreadable status is a catalog defect whatever the
        # licence values beside it say. A null status and an absent key are the
        # same defect: the row states nothing about how far its licence reading
        # got. The isinstance check keeps a catalog that records a list or an
        # object there as data rather than as a TypeError out of a membership
        # test. Nothing read this field until 2026-09-12, so every one of those
        # cases cleared.
        if not isinstance(code_status, str) or code_status not in code_status_vocabulary:
            commercial_problems.append(
                _problem(
                    "error",
                    "LICENCE_STATUS_UNRECOGNIZED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the code licence's status is "
                    f"{code_status!r}, which is not one of the recognized values "
                    f"{sorted(code_status_vocabulary)}; refusing a commercial campaign "
                    "rather than reading an unrecognized licence status as a settled "
                    f"licence record. Correct catalog.tools.{tool_id}.code_licence.status, "
                    f"or remove '{tool_id}' via {selection['selected_via']}.",
                    source=(
                        f"{_citation(entry, 'code_licence')}; "
                        f"catalog.{CODE_LICENCE_STATUS_ENUM_KEY}"
                    ),
                )
            )

        if needs_agreement and commercial_value != "permitted_with_conditions":
            commercial_problems.append(
                _problem(
                    "error",
                    "CATALOG_GATE_CONFLICT",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': catalog commercial_use is "
                    f"{commercial_value!r}, but the gate declares a written-agreement "
                    "exception; use permitted_with_conditions only when a separately "
                    "executed agreement can authorize commercial use",
                    source=(
                        f"{_citation(entry, 'code_licence')}; "
                        f"catalog.tools.{tool_id}.gate.requires_written_agreement_for_commercial"
                    ),
                )
            )
        elif needs_agreement:
            reference = _agreement_reference(config, tool_id)
            if reference is None:
                commercial_problems.append(
                    _problem(
                        "error",
                        "WRITTEN_AGREEMENT_REQUIRED",
                        tool_id,
                        f"'{entry.get('display_name', tool_id)}': commercial use requires "
                        f"a separate written licence agreement ({_citation(entry, 'code_licence')}). "
                        f"Obtain one and record it at {AGREEMENT_SECTION_KEY}."
                        f"{tool_id}.{AGREEMENT_FIELD}, or remove '{tool_id}' via "
                        f"{selection['selected_via']}.",
                        source=(
                            f"{_citation(entry, 'code_licence')}; "
                            f"configuration.{AGREEMENT_SECTION_KEY}.{tool_id}.{AGREEMENT_FIELD}"
                        ),
                    )
                )
            else:
                commercial_warnings.append(
                    _problem(
                        "warning",
                        "WRITTEN_AGREEMENT_RECORDED",
                        tool_id,
                        f"'{entry.get('display_name', tool_id)}': agreement reference "
                        f"'{reference}' recorded; confirm its scope covers this campaign's "
                        "use before relying on it",
                        source=f"configuration.{AGREEMENT_SECTION_KEY}.{tool_id}.{AGREEMENT_FIELD}",
                    )
                )
        elif commercial_value == "forbidden":
            commercial_problems.append(
                _problem(
                    "error",
                    "COMMERCIAL_USE_FORBIDDEN",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': licence forbids the declared "
                    f"use ({_citation(entry, 'code_licence')})",
                    source=_citation(entry, "code_licence"),
                )
            )
        elif commercial_value in UNSETTLED_COMMERCIAL_VALUES:
            commercial_problems.append(
                _problem(
                    "error",
                    "LICENCE_UNKNOWN_COMMERCIAL_USE",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': commercial use is unsettled "
                    f"in the catalog ({_citation(entry, 'code_licence')}); refusing a "
                    "commercial campaign until the catalog cites a source. Remove "
                    f"'{tool_id}' via {selection['selected_via']} to run without it.",
                    source=_citation(entry, "code_licence"),
                )
            )
        elif commercial_value not in vocabulary:
            commercial_problems.append(
                _problem(
                    "error",
                    "COMMERCIAL_USE_UNRECOGNIZED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': catalog commercial_use is "
                    f"{commercial_value!r}, which is not one of the recognized values "
                    f"{sorted(vocabulary)}; refusing a commercial campaign rather than "
                    "reading an unrecognized licence value as permission. Correct "
                    f"catalog.tools.{tool_id}.code_licence.commercial_use, or remove "
                    f"'{tool_id}' via {selection['selected_via']}.",
                    source=(
                        f"{_citation(entry, 'code_licence')}; "
                        f"catalog.{COMMERCIAL_USE_ENUM_KEY}"
                    ),
                )
            )
        elif commercial_value == "permitted_with_conditions" and not conditions:
            commercial_problems.append(
                _problem(
                    "error",
                    "LICENCE_CONDITION_UNSTATED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the catalog records commercial "
                    "use as permitted_with_conditions and states no condition "
                    f"({_citation(entry, 'code_licence')}); refusing a commercial campaign "
                    "because the gate cannot tell the scientist what to comply with. Record "
                    f"each condition at catalog.tools.{tool_id}.gate.conditions, or remove "
                    f"'{tool_id}' via {selection['selected_via']}.",
                    source=f"catalog.tools.{tool_id}.gate.conditions",
                )
            )
        elif isinstance(code_status, str) and code_status in OPEN_CODE_LICENCE_RECORD:
            # The licence value is settled here and permits the declared use.
            # What is open is the reading behind it: the row's own status says
            # nobody finished opening the licence this entry names. Calling that
            # a prohibition would tell a scientist to drop a tool that may well
            # be theirs to run, so it carries its own code.
            commercial_problems.append(
                _problem(
                    "error",
                    "LICENCE_PROVENANCE_UNVERIFIED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the code licence record is "
                    f"unsettled. The catalog records commercial use as {commercial_value!r} "
                    f"and records the licence row itself as {code_status!r} "
                    f"({_citation(entry, 'code_licence')}). Refusing a commercial campaign "
                    "until the row states which licence text was read: cite it, or close "
                    "the reading the entry's evidence names. Remove "
                    f"'{tool_id}' via {selection['selected_via']} to run without it.",
                    source=_citation(entry, "code_licence"),
                )
            )

        weight_status = weights.get("status")
        weight_commercial = weights.get("commercial_use")
        # The provenance status is checked against the declared vocabulary on its
        # own, because an unreadable status is a catalog defect whatever the
        # licence values beside it say. A null status and an absent key are the
        # same defect: the row states nothing about how far its weights reading
        # got. This refusal covers both, so the provenance branch further down
        # never has to guess what an unrecognized spelling meant. The isinstance
        # check keeps a catalog that records a list or an object there as data
        # rather than as a TypeError out of a membership test.
        if not isinstance(weight_status, str) or weight_status not in status_vocabulary:
            commercial_problems.append(
                _problem(
                    "error",
                    "WEIGHTS_STATUS_UNRECOGNIZED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the model weights' status is "
                    f"{weight_status!r}, which is not one of the recognized values "
                    f"{sorted(status_vocabulary)}; refusing a commercial campaign rather "
                    "than reading an unrecognized provenance status as a settled weight "
                    f"record. Correct catalog.tools.{tool_id}.weights.status, or remove "
                    f"'{tool_id}' via {selection['selected_via']}.",
                    source=(
                        f"{_citation(entry, 'weights')}; catalog.{WEIGHTS_STATUS_ENUM_KEY}"
                    ),
                )
            )
        if weight_commercial == "forbidden":
            commercial_problems.append(
                _problem(
                    "error",
                    "WEIGHTS_COMMERCIAL_USE_FORBIDDEN",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the model weights' licence "
                    f"forbids the declared use ({_citation(entry, 'weights')}). No agreement "
                    f"reference lifts this. Remove '{tool_id}' via "
                    f"{selection['selected_via']}, or select weights whose licence permits "
                    "commercial use.",
                    source=_citation(entry, "weights"),
                )
            )
        elif weight_commercial in UNSETTLED_COMMERCIAL_VALUES:
            commercial_problems.append(
                _problem(
                    "error",
                    "WEIGHTS_COMMERCIAL_USE_UNKNOWN",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the model weights' licence "
                    f"is unresolved ({_citation(entry, 'weights')}); refusing a commercial "
                    "campaign until the governing weight-set terms are cited. Remove "
                    f"'{tool_id}' via {selection['selected_via']} to run without it.",
                    source=_citation(entry, "weights"),
                )
            )
        elif weight_commercial not in vocabulary | {NOT_APPLICABLE}:
            commercial_problems.append(
                _problem(
                    "error",
                    "WEIGHTS_COMMERCIAL_USE_UNRECOGNIZED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the model weights' "
                    f"commercial_use is {weight_commercial!r}, which is not one of the "
                    f"recognized values {sorted(vocabulary | {NOT_APPLICABLE})}; refusing a "
                    "commercial campaign rather than reading an unrecognized licence value "
                    f"as permission. Correct catalog.tools.{tool_id}.weights.commercial_use, "
                    f"or remove '{tool_id}' via {selection['selected_via']}.",
                    source=(
                        f"{_citation(entry, 'weights')}; catalog.{COMMERCIAL_USE_ENUM_KEY}"
                    ),
                )
            )
        elif weight_commercial == "permitted_with_conditions" and not conditions:
            commercial_problems.append(
                _problem(
                    "error",
                    "WEIGHTS_CONDITION_UNSTATED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the catalog records the "
                    "weights as permitted_with_conditions and states no condition "
                    f"({_citation(entry, 'weights')}); refusing a commercial campaign "
                    "because the gate cannot tell the scientist what to comply with. "
                    f"Record each condition at catalog.tools.{tool_id}.gate.conditions, or "
                    f"remove '{tool_id}' via {selection['selected_via']}.",
                    source=f"catalog.tools.{tool_id}.gate.conditions",
                )
            )
        elif (
            weight_commercial != NOT_APPLICABLE
            and isinstance(weight_status, str)
            and weight_status in OPEN_WEIGHT_PROVENANCE
        ):
            # The licence is settled here and permits the declared use. What is open
            # is provenance: the row's own status says its weights reading is not
            # finished. Calling that a licence problem sends a scientist to read
            # terms that already answer the question they have, so it carries its
            # own code. An N/A row names no artifact, so this refusal cannot apply
            # to one.
            commercial_problems.append(
                _problem(
                    "error",
                    "WEIGHTS_PROVENANCE_UNVERIFIED",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the model weights' "
                    "provenance is unsettled. The weights licence "
                    f"({weights.get('spdx')}) permits commercial use, and "
                    f"the catalog records the weights row itself as {weight_status!r} "
                    f"({_citation(entry, 'weights')}). Refusing a commercial campaign "
                    "until the row states which artifact would run: record the revision "
                    "or SHA-256, or close the weight-terms reading the entry's note "
                    f"names. Remove '{tool_id}' via {selection['selected_via']} to run "
                    "without it.",
                    source=_citation(entry, "weights"),
                )
            )

        if weight_commercial == "permitted_with_conditions" and conditions:
            # The pass is conditional, so the report says which layer carries the
            # condition. A scientist who reads only the tool's licence row would
            # otherwise see a clear gate and no obligation.
            commercial_warnings.append(
                _problem(
                    "warning",
                    "WEIGHTS_COMMERCIAL_USE_CONDITIONAL",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': the model weights permit "
                    "commercial use under stated conditions "
                    f"({_citation(entry, 'weights')}). Satisfy each of them before relying "
                    "on this pass: " + "; ".join(conditions),
                    source=f"catalog.tools.{tool_id}.gate.conditions",
                )
            )

        for condition in entry.get("gate", {}).get("conditions", []) or []:
            commercial_warnings.append(
                _problem(
                    "warning",
                    "LICENCE_CONDITION",
                    tool_id,
                    f"'{entry.get('display_name', tool_id)}': {condition}",
                    source=_citation(entry, "gate"),
                )
            )

    if use == "unknown":
        # The campaign is still refused, because DECLARED_USE_REQUIRED is an error
        # and nothing here lowers it. What changes is that the commercial findings
        # arrive as the priced consequence of one answer rather than as separate
        # faults, so the scientist reads the question and the cost of each answer
        # in one place. Nothing is dropped: every finding keeps its message,
        # citation and tool, under a code that says the condition it depends on.
        blocked = sorted(
            {finding["tool_id"] for finding in commercial_problems if finding["tool_id"]}
        )
        for finding in commercial_problems + commercial_warnings:
            preview = dict(finding)
            preview["severity"] = "warning"
            preview["code"] = f"{finding['code']}_IF_COMMERCIAL"
            preview["message"] = (
                'if declared_use is set to "commercial": ' + finding["message"]
            )
            warnings.append(preview)
        problems.insert(
            0,
            _problem(
                "error",
                "DECLARED_USE_REQUIRED",
                None,
                f'configuration key "{DECLARED_USE_KEY}" is unset or unrecognized, so this '
                'campaign is evaluated conservatively as commercial. Set it to '
                '"non-commercial" or "commercial". "non-commercial" checks only that every '
                'selected tool has a catalog entry. "commercial" additionally requires cited '
                "commercial-use terms for each tool's code and weights, which "
                + (
                    f"{len(blocked)} of the {len(checked)} selected tools do not have: "
                    + ", ".join(blocked)
                    if blocked
                    else f"all {len(checked)} selected tools already have"
                )
                + ". The warnings on this report preview every commercial finding for this "
                "exact tool selection.",
                source=f"configuration.{DECLARED_USE_KEY}",
            ),
        )
    else:
        problems.extend(commercial_problems)
        warnings.extend(commercial_warnings)

    errors = [p for p in problems if p["severity"] == "error"]
    refusals = [_refusal_for_problem(problem) for problem in errors]
    for problem, refusal in zip(errors, refusals):
        problem["detail"] = problem["message"]
        problem["refusal"] = refusal.as_dict()
        problem["message"] = refusal.text()
    return {
        "gate": "claude-binder.tool-licence-gate.v0",
        "ok": not errors,
        "declared_use": use,
        "commercial_risk_evaluated": commercial_risk,
        "checked_tools": checked,
        "problems": problems,
        "warnings": warnings,
        "exit_code": exit_code_for_result(verified=not errors, refused=bool(errors)),
        "refusals": [refusal.as_dict() for refusal in refusals],
        "refusal_text": "\n\n".join(refusal.text() for refusal in refusals),
    }


# A minimal catalog used only by --self-test. Shape-identical to catalog.json,
# including all three vocabularies the gate reads and the N/A weights value every
# no-weights row in catalog.json carries.
CATALOG_FIXTURE = {
    "commercial_use_enum": list(GATE_COMMERCIAL_USE_VALUES),
    "code_licence_status_enum": list(GATE_CODE_LICENCE_STATUS_VALUES),
    "weights_status_enum": list(GATE_WEIGHTS_STATUS_VALUES),
    "tools": {
        "chimerax": {
            "display_name": "UCSF ChimeraX",
            "code_licence": {
                "commercial_use": "permitted_with_conditions",
                "status": "resolved",
                "evidence": [
                    "references/tool-licences.md:23 | No. Commercial use requires a separate written licence agreement"
                ],
            },
            "weights": {"status": "not_applicable", "commercial_use": NOT_APPLICABLE},
            "gate": {
                "requires_written_agreement_for_commercial": True,
                "conditions": [],
            },
        },
        "genie3": {
            "display_name": "Genie3",
            "code_licence": {
                "commercial_use": "permitted",
                "status": "resolved",
                "evidence": ["references/tool-licences.md:12 | Apache-2.0 for code"],
            },
            "weights": {
                "spdx": "Apache-2.0",
                "status": "resolved",
                "commercial_use": "permitted",
                "evidence": [
                    "references/tool-licences.md:12 | Apache-2.0, declared by the yeqinglin/genie3 model card"
                ],
            },
            "gate": {},
        },
        # The unknown-weights case. Its values are the ones catalog.json records:
        # the code licence is BSD-3-Clause with no commercial-use sentence, and no
        # publisher source applies any licence to the checkpoint.
        "rfdiffusion3": {
            "display_name": "RFdiffusion3",
            "code_licence": {
                "commercial_use": "unknown",
                "status": "resolved_licence_unknown_commercial",
                "evidence": [
                    "references/tool-licences.md | RFdiffusion3 row: BSD-3-Clause for code, and no express commercial-use sentence"
                ],
            },
            "weights": {
                "status": "unverified",
                "commercial_use": "unknown",
                "evidence": [
                    "references/tool-licences.md | RFdiffusion3 row: weights licence MISSING, no publisher source licences the checkpoint"
                ],
            },
            "gate": {},
        },
        "pymol-open-source": {
            "display_name": "PyMOL (open-source)",
            "code_licence": {
                "commercial_use": "permitted",
                "status": "resolved",
                "evidence": ["references/tool-licences.md:22 | permitted for any purpose"],
            },
            "weights": {"status": "not_applicable", "commercial_use": NOT_APPLICABLE},
            "gate": {},
        },
        # The forbidden-weights case. No catalog.json row carries it today, and
        # the gate cleared it silently until 2026-09-12, so the fixture holds one.
        "forbidden-weights-example": {
            "display_name": "Forbidden-weights example",
            "code_licence": {
                "commercial_use": "permitted",
                "status": "resolved",
                "evidence": ["fixture | code licence permits commercial use"],
            },
            "weights": {
                "status": "resolved",
                "commercial_use": "forbidden",
                "evidence": ["fixture | the weight terms forbid commercial use"],
            },
            "gate": {},
        },
        "boltz": {
            "display_name": "Boltz",
            "code_licence": {
                "commercial_use": "permitted",
                "status": "resolved",
                "evidence": ["references/tool-licences.md:15 | MIT"],
            },
            "weights": {"status": "resolved", "commercial_use": "permitted"},
            "gate": {},
        },
        # The open-provenance case, shaped on catalog.json's protenix-v2 row: the
        # weights licence permits the use, and the row's own status says the
        # bytes are not pinned. It refuses on provenance while its licence
        # permits the campaign.
        "open-provenance-example": {
            "display_name": "Open-provenance example",
            "code_licence": {
                "commercial_use": "permitted",
                "status": "resolved",
                "evidence": ["fixture | code licence permits commercial use"],
            },
            "weights": {
                "spdx": "Apache-2.0",
                "status": "open_todo",
                "commercial_use": "permitted",
                "evidence": ["fixture | the weight terms permit commercial use"],
            },
            "gate": {},
        },
        # The open code-licence case, shaped on catalog.json's pxdesign row: the
        # recorded commercial-use value permits the campaign, and the row's own
        # status says nobody finished reading the licence behind it. It refuses on
        # the licence record while its licence value permits the campaign. Nothing
        # read code_licence.status until 2026-09-12, so this row cleared.
        "open-code-licence-example": {
            "display_name": "Open-code-licence example",
            "code_licence": {
                "commercial_use": "permitted",
                "status": "unverified",
                "evidence": ["fixture | the pinned commit is recorded and the term is not"],
            },
            "weights": {"status": "not_applicable", "commercial_use": NOT_APPLICABLE},
            "gate": {},
        },
    }
}


def _self_test() -> int:
    catalog = {
        "commercial_use_enum": list(CATALOG_FIXTURE["commercial_use_enum"]),
        "code_licence_status_enum": list(CATALOG_FIXTURE["code_licence_status_enum"]),
        "weights_status_enum": list(CATALOG_FIXTURE["weights_status_enum"]),
        "tools": {k: v for k, v in CATALOG_FIXTURE["tools"].items() if v is not None},
    }

    def run(name: str, config: dict[str, Any]) -> bool:
        report = evaluate(config, catalog)
        codes = [
            p["code"] for p in report["problems"] if p["severity"] == "error"
        ]
        warn_codes = [w["code"] for w in report["warnings"]]
        print(f"[{name}] ok={report['ok']} errors={codes} warnings={warn_codes}")
        return report

    base = {
        "generation": {"generators": []},
        "sequence_design": {"designers": []},
        "cofold": {"predictors": []},
        "adapters": [
            {
                "adapter_id": "viewer-renderer",
                "role": "renderer",
                "command_argv_template": ["{{python_executable}}", "--pymol", "pymol"],
            }
        ],
    }

    chimerax = {
        **base,
        "cofold": {"predictors": [{"id": "chimerax", "enabled": True}]},
    }

    # 1. Worked case: commercial campaign rendering through ChimeraX.
    r1 = run("chimerax-commercial-no-agreement", {**chimerax, DECLARED_USE_KEY: "commercial"})
    assert not r1["ok"]
    assert any(p["code"] == "WRITTEN_AGREEMENT_REQUIRED" and p["tool_id"] == "chimerax" for p in r1["problems"])

    # 1b. Same campaign with a recorded agreement reference: passes with warning.
    r1b = run(
        "chimerax-commercial-with-agreement",
        {
            **chimerax,
            DECLARED_USE_KEY: "commercial",
            AGREEMENT_SECTION_KEY: {"chimerax": {AGREEMENT_FIELD: "UCSF-EXAMPLE-001"}},
        },
    )
    assert r1b["ok"]
    assert any(w["code"] == "WRITTEN_AGREEMENT_RECORDED" for w in r1b["warnings"])

    # 2. Unknown weights licence blocks a commercial campaign. RFdiffusion3 is the
    # tool that genuinely holds one: no publisher source licences its checkpoint.
    # Its code licence is unsettled too, so two errors fire and the assertion pins
    # the weights one.
    r2 = run(
        "rfdiffusion3-commercial",
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {"generators": [{"id": "rfdiffusion3", "enabled": True}]},
        },
    )
    assert not r2["ok"]
    assert any(p["code"] == "WEIGHTS_COMMERCIAL_USE_UNKNOWN" for p in r2["problems"])

    # 2b. A settled weights licence clears the same campaign. Genie3's weights read
    # MISSING from a GitHub README until the model card at the weights host was
    # opened, and the gate refused a tool whose licence permits the use.
    r2b = run(
        "genie3-commercial-clears",
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {"generators": [{"id": "genie3", "enabled": True}]},
        },
    )
    assert r2b["ok"]
    assert not r2b["problems"]

    # 2c. Weights whose licence forbids commercial use refuse the campaign. The
    # weights check was a denylist of unsettled values until 2026-09-12, so this
    # value cleared the gate.
    r2c = run(
        "forbidden-weights-commercial",
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {"generators": [{"id": "forbidden-weights-example", "enabled": True}]},
        },
    )
    assert not r2c["ok"]
    assert any(p["code"] == "WEIGHTS_COMMERCIAL_USE_FORBIDDEN" for p in r2c["problems"])

    # 2d. A value outside the catalog's declared vocabulary refuses. A typo in a
    # licence field must not read as permission.
    typo_catalog = {
        "commercial_use_enum": catalog["commercial_use_enum"],
        "code_licence_status_enum": catalog["code_licence_status_enum"],
        "weights_status_enum": catalog["weights_status_enum"],
        "tools": {
            **catalog["tools"],
            "genie3": {
                **catalog["tools"]["genie3"],
                "weights": {**catalog["tools"]["genie3"]["weights"], "commercial_use": "perrmited"},
            },
        },
    }
    config_2d = {
        **base,
        DECLARED_USE_KEY: "commercial",
        "generation": {"generators": [{"id": "genie3", "enabled": True}]},
    }
    r2d = evaluate(config_2d, typo_catalog)
    print(
        "[unrecognized-weights-value] ok="
        f"{r2d['ok']} errors={[p['code'] for p in r2d['problems']]}"
    )
    assert not r2d["ok"]
    assert any(p["code"] == "WEIGHTS_COMMERCIAL_USE_UNRECOGNIZED" for p in r2d["problems"])

    # 2e. A permitting weights licence beside an unfinished weights record refuses
    # on provenance, under a code that does not read as a licence prohibition.
    r2e = run(
        "open-provenance-commercial",
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {"generators": [{"id": "open-provenance-example", "enabled": True}]},
        },
    )
    assert not r2e["ok"]
    assert any(p["code"] == "WEIGHTS_PROVENANCE_UNVERIFIED" for p in r2e["problems"])
    assert not any(p["code"] == "WEIGHTS_COMMERCIAL_USE_UNKNOWN" for p in r2e["problems"])

    # 2f. A status outside the declared vocabulary refuses. The check was a
    # denylist of four spellings until 2026-09-12, so "unverified" cleared it.
    unknown_status_catalog = {
        "commercial_use_enum": catalog["commercial_use_enum"],
        "code_licence_status_enum": catalog["code_licence_status_enum"],
        "weights_status_enum": catalog["weights_status_enum"],
        "tools": {
            **catalog["tools"],
            "genie3": {
                **catalog["tools"]["genie3"],
                "weights": {**catalog["tools"]["genie3"]["weights"], "status": "wibble"},
            },
        },
    }
    r2f = evaluate(
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {"generators": [{"id": "genie3", "enabled": True}]},
        },
        unknown_status_catalog,
    )
    print(
        "[unrecognized-weights-status] ok="
        f"{r2f['ok']} errors={[p['code'] for p in r2f['problems']]}"
    )
    assert not r2f["ok"]
    assert any(p["code"] == "WEIGHTS_STATUS_UNRECOGNIZED" for p in r2f["problems"])

    # 2g. A permitting commercial-use value beside an unread licence record
    # refuses on the record, under a code that does not read as a prohibition.
    # code_licence.status was read by nothing until 2026-09-12.
    r2g = run(
        "open-code-licence-commercial",
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {
                "generators": [{"id": "open-code-licence-example", "enabled": True}]
            },
        },
    )
    assert not r2g["ok"]
    assert any(p["code"] == "LICENCE_PROVENANCE_UNVERIFIED" for p in r2g["problems"])
    assert not any(p["code"] == "COMMERCIAL_USE_FORBIDDEN" for p in r2g["problems"])

    # 2h. A code-licence status outside the vocabulary refuses as a catalog
    # defect rather than as an unread record, because an unreadable status says
    # nothing about how far the reading got.
    r2h = evaluate(
        {
            **base,
            DECLARED_USE_KEY: "commercial",
            "generation": {"generators": [{"id": "genie3", "enabled": True}]},
        },
        {
            **catalog,
            "tools": {
                **catalog["tools"],
                "genie3": {
                    **catalog["tools"]["genie3"],
                    "code_licence": {
                        **catalog["tools"]["genie3"]["code_licence"],
                        "status": "wibble",
                    },
                },
            },
        },
    )
    print(
        "[unrecognized-code-licence-status] ok="
        f"{r2h['ok']} errors={[p['code'] for p in r2h['problems']]}"
    )
    assert not r2h["ok"]
    assert any(p["code"] == "LICENCE_STATUS_UNRECOGNIZED" for p in r2h["problems"])

    # 2i to 2k. The catalog is the document this gate polices, so widening one of
    # its vocabularies must not widen what clears. Each scenario declares a
    # nonsense value in one enum, writes it into the field that enum governs, and
    # expects the same refusal the undeclared value draws. Every one of the three
    # cleared until 2026-09-12.
    widening_cases = (
        (
            "widened-commercial-use-enum",
            COMMERCIAL_USE_ENUM_KEY,
            "perrmited",
            "code_licence",
            "commercial_use",
            "COMMERCIAL_USE_UNRECOGNIZED",
        ),
        (
            "widened-code-licence-status-enum",
            CODE_LICENCE_STATUS_ENUM_KEY,
            "totally_fine",
            "code_licence",
            "status",
            "LICENCE_STATUS_UNRECOGNIZED",
        ),
        (
            "widened-weights-status-enum",
            WEIGHTS_STATUS_ENUM_KEY,
            "totally_fine",
            "weights",
            "status",
            "WEIGHTS_STATUS_UNRECOGNIZED",
        ),
    )
    for name, enum_key, invented, block, field, expected_code in widening_cases:
        widened = {
            **catalog,
            enum_key: list(catalog[enum_key]) + [invented],
            "tools": {
                **catalog["tools"],
                "genie3": {
                    **catalog["tools"]["genie3"],
                    block: {**catalog["tools"]["genie3"][block], field: invented},
                },
            },
        }
        widened_report = evaluate(
            {
                **base,
                DECLARED_USE_KEY: "commercial",
                "generation": {"generators": [{"id": "genie3", "enabled": True}]},
            },
            widened,
        )
        print(
            f"[{name}] ok={widened_report['ok']} "
            f"errors={[p['code'] for p in widened_report['problems']]}"
        )
        assert not widened_report["ok"], name
        assert any(p["code"] == expected_code for p in widened_report["problems"]), name

    # 3. Non-commercial campaign using the non-commercial tool is fine.
    r3 = run("chimerax-non-commercial", {**chimerax, DECLARED_USE_KEY: "non-commercial"})
    assert r3["ok"]

    # 4. Uncatalogued tool fails closed either way.
    r4 = run(
        "uncatalogued-tool",
        {
            **base,
            DECLARED_USE_KEY: "non-commercial",
            "cofold": {"predictors": [{"id": "neverheardofit", "enabled": True}]},
        },
    )
    assert not r4["ok"]
    assert any(p["code"] == "TOOL_NOT_IN_CATALOG" for p in r4["problems"])

    # 5. An undeclared use refuses on the one decision and previews the rest.
    r5 = run(
        "undeclared-use-previews-commercial",
        {**base, "generation": {"generators": [{"id": "rfdiffusion3", "enabled": True}]}},
    )
    assert not r5["ok"]
    assert [p["code"] for p in r5["problems"]] == ["DECLARED_USE_REQUIRED"]
    assert any(
        w["code"] == "WEIGHTS_COMMERCIAL_USE_UNKNOWN_IF_COMMERCIAL" for w in r5["warnings"]
    )

    print("self-test passed: 16 scenarios")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="composed campaign/profile JSON to check")
    parser.add_argument("--catalog", help="tool catalog JSON (e.g. catalog.json)")
    parser.add_argument("--declared-use", choices=["commercial", "non-commercial"])
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--self-test", action="store_true", help="run built-in scenarios")
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.config or not args.catalog:
        parser.error("--config and --catalog are required unless --self-test is given")

    try:
        config = json.loads(open(args.config, "r", encoding="utf-8").read())
        catalog = load_catalog(args.catalog)
        report = evaluate(config, catalog, declared_use=args.declared_use)
    except (GateError, OSError, ValueError, json.JSONDecodeError) as exc:
        refusal = Refusal(
            cause="The licence gate could not read its inputs.",
            expected="Readable configuration and catalog JSON files.",
            expected_source="The --config and --catalog arguments.",
            found=f"{type(exc).__name__}: {exc}",
            found_source="The file read that raised the error.",
            scope="The licence gate started no provider command.",
            action="Supply readable configuration and catalog JSON files.",
            escalation="Send the input file error to the campaign maintainer.",
        )
        print(refusal.text(), file=sys.stderr)
        return refusal.exit_code
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        print(report["refusal_text"], file=sys.stderr)
    return int(ExitCode.VERIFIED) if report["ok"] else int(ExitCode.PREFLIGHT_REFUSAL)


if __name__ == "__main__":
    sys.exit(main())
