"""Resolve campaign tool choices before a campaign starts a paid stage.

The menu reads a composed configuration and supplied evidence. It does not run a
toolcheck, probe a provider, read credentials, or launch a subprocess. Callers
can pass a cached authorization report from ``provider_authorization`` and free
runtime-check results that they already collected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import arms, gate, platform_inventory, provider_authorization
from .paths import package_file


MENU_SCHEMA_VERSION = 1
CATALOG_MENU_SCHEMA_VERSION = 1
_TOKEN_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_FIXTURE_MODULE = "claude_binder.fixture_adapter"
_STDLIB_ENVIRONMENT = "standard-library-python"
_RETIRED_FILTERS = {
    "structure_novelty": "Structural novelty needs a pinned TM-align executable.",
    "secondary_structure": "Secondary structure needs a DSSP executable.",
}
_DISPLAY_NAMES = {
    "rfdiffusion": "RFdiffusion",
    "rfdiffusion3": "RFdiffusion3",
    "genie3": "Genie3",
    "proteinmpnn": "ProteinMPNN",
    "solublempnn": "SolubleMPNN",
    "ligandmpnn": "LigandMPNN",
    "fair-esm2": "ESM-2",
    "esmfold2": "ESMFold2-Full",
    "esmfold2-fast": "ESMFold2-Fast",
    "esmfold2-native-design": "ESMFold2-Native-Design (Experimental)",
    "protenix-v2": "Protenix v2",
    "alphafold-multimer-v3": "AlphaFold2-Multimer-v3",
    "boltz": "Boltz",
    "chai1": "Chai-1",
    "openfold3": "OpenFold3",
    "diffdock": "DiffDock-L",
    "ncbi-sequence-fetch": "NCBI sequence fetch",
    "ipsae": "ipSAE",
    "dockq": "DockQ",
    "mmseqs2": "MMseqs2",
    "uniref90": "UniRef90",
    "colabfold-msa-server": "ColabFold MSA server (api.colabfold.com)",
    "pymol-open-source": "PyMOL (open-source)",
    "chimerax": "UCSF ChimeraX",
}
_AUXILIARY_DISPLAY_NAMES = {
    "browser-viewer": "Browser viewer",
    "viewer-renderer": "PyMOL figures",
    "structure-picture-renderer": "Overview and interface PNG set",
    "composition": "Composition",
    "exact_duplicates": "Exact duplicates",
    "liability_chemistry": "Liability chemistry",
    "sequence_novelty": "Sequence novelty",
    "model_likelihood": "Model likelihood",
    "structure_novelty": "Structural novelty",
    "secondary_structure": "Secondary structure",
}
_FILTER_STAGE_DEFAULTS = {
    "composition": "filter-integrity",
    "exact_duplicates": "filter-integrity",
    "liability_chemistry": "filter-integrity",
    "sequence_novelty": "filter-novelty",
    "model_likelihood": "filter-novelty",
}

_WORKFLOW_ROUTES = [
    {
        "id": "connector-authoring",
        "use_when": "Connect a selected service or tool, inspect an inherited adapter, or prepare a declared substitution.",
        "reference": "references/connector-authoring.md",
        "inspection_module": "claude_binder.connector_contract",
        "profiles": {},
        "rules": [
            "Resolve the selected profile before comparing toolcheck, run, and parse contracts.",
            "Match the service request and returned artifacts before classifying a route as qualification-only.",
            "Keep model identity, route readiness, and scientific qualification as separate evidence.",
        ],
    },
    {
        "id": "new-target-provider-canary",
        "use_when": (
            "Prepare a new target and compare a small supplied-candidate panel across "
            "one or more execution routes before a wider campaign."
        ),
        "reference": "references/new-target-provider-canary.md",
        "profiles": {
            "fal": "supplied-candidates-fal.template.json",
            "modal": "supplied-candidates-modal.template.json",
        },
        "rules": [
            "Use one target dossier and one candidate/control manifest for every route.",
            "Call a result same-model only when checkpoint revision, MSA policy, inference settings, sequences, chain convention, and seed match.",
            "Run one real prediction, parse its declared artifacts, and close or reconcile it before widening.",
            "Use the packaged adapters and guarded dispatcher rather than scripts recovered from historical evidence directories.",
        ],
    }
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _named_hosts(value: object) -> list[str]:
    """Return the job-time hosts a catalogue row names, flat or keyed by route."""
    if isinstance(value, Mapping):
        named: set[str] = set()
        for entry in value.values():
            named.update(_named_hosts(entry))
        return sorted(named)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def _display_name(identifier: str) -> str:
    """Return a recorded scientist-facing name or the declared identifier."""
    return _DISPLAY_NAMES.get(identifier, _AUXILIARY_DISPLAY_NAMES.get(identifier, identifier))


def _enabled_items(config: Mapping[str, Any], section: str, key: str) -> list[Mapping[str, Any]]:
    return [
        item
        for item in _mapping_list(_mapping(config.get(section)).get(key))
        if item.get("enabled") is not False
    ]


def _adapter_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        adapter_id: adapter
        for adapter in _mapping_list(config.get("adapters"))
        if isinstance((adapter_id := adapter.get("adapter_id")), str) and adapter_id
    }


def _stage_adapter_map(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        stage_id: adapter_id
        for stage in _mapping_list(config.get("stages"))
        if isinstance((stage_id := stage.get("stage_id")), str)
        and stage_id
        and isinstance((adapter_id := stage.get("adapter_id")), str)
        and adapter_id
    }


def _adapter_tokens(adapter: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("toolcheck_argv", "command_argv_template", "parser_argv_template"):
        for token in _mapping_tokens(adapter.get(key)):
            tokens.add(token)
    return tokens


def _mapping_tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(_TOKEN_RE.findall(value))
    if isinstance(value, Mapping):
        return {token for child in value.values() for token in _mapping_tokens(child)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {token for child in value for token in _mapping_tokens(child)}
    return set()


def _has_fixture_route(adapter: Mapping[str, Any]) -> bool:
    return _FIXTURE_MODULE in " ".join(
        str(value)
        for key in ("toolcheck_argv", "command_argv_template", "parser_argv_template")
        for value in _mapping_list_values(adapter.get(key))
    )


def _mapping_list_values(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _is_stdlib_route(adapter: Mapping[str, Any]) -> bool:
    if adapter.get("environment_identity") != _STDLIB_ENVIRONMENT:
        return False
    command = adapter.get("command_argv_template")
    values = _mapping_list_values(command)
    return any(isinstance(value, str) and value.startswith("claude_binder") for value in values)


def _provider_statuses(provider_report: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(provider_report, Mapping):
        return {}
    statuses: dict[str, str] = {}
    for provider_id in ("modal", "runpod", "lambda"):
        authorization = provider_report.get(f"{provider_id}_authorization")
        if isinstance(authorization, Mapping):
            status = authorization.get("status")
            if isinstance(status, str):
                statuses[provider_id] = status
    direct_provider = provider_report.get("provider")
    direct_status = provider_report.get("status")
    if isinstance(direct_provider, str) and isinstance(direct_status, str):
        statuses[direct_provider] = direct_status
    applications = provider_report.get("applications")
    if isinstance(applications, Sequence) and not isinstance(applications, (str, bytes)):
        for application in applications:
            if not isinstance(application, Mapping):
                continue
            field = application.get("endpoint_field")
            status = application.get("status")
            if isinstance(field, str) and isinstance(status, str):
                statuses[field] = status
        return statuses
    statuses.update(
        {
            str(field): str(status)
            for field, status in provider_report.items()
            if isinstance(status, str) and field not in {"status", "provider"}
        }
    )
    return statuses


def _native_provider(adapter: Mapping[str, Any]) -> str | None:
    environment = adapter.get("environment")
    if not isinstance(environment, Mapping):
        return None
    return {
        "modal-platform": "modal",
        "runpod-platform": "runpod",
        "lambda-platform": "lambda",
    }.get(environment.get("CLAUDE_BINDER_EXECUTION_ROUTE"))


def _runtime_status(runtime_checks: Mapping[str, Any] | None, adapter_id: str) -> str | None:
    if not isinstance(runtime_checks, Mapping):
        return None
    value = runtime_checks.get(adapter_id)
    if isinstance(value, Mapping):
        value = value.get("status")
    if value is True or value == "pass":
        return "pass"
    if value is False or value == "fail" or value == "could not be checked":
        return "fail"
    return None


def _platform_status(platform_tools: Mapping[str, Any] | None, adapter: Mapping[str, Any]) -> str | None:
    platform_tool = adapter.get("platform_tool")
    if not isinstance(platform_tool, str) or not platform_tool:
        return None
    if not isinstance(platform_tools, Mapping):
        return "unknown"
    value = platform_tools.get(platform_tool)
    if isinstance(value, Mapping):
        value = value.get("status")
    if value is True or value == "available":
        return "available"
    if value is False or value == "unavailable":
        return "unavailable"
    return "unknown"


def _valid_endpoint(endpoint: Any) -> bool:
    return provider_authorization.queue_status_url(endpoint) is not None


def _availability(
    adapter_id: str | None,
    adapter: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    *,
    provider_statuses: Mapping[str, str],
    runtime_checks: Mapping[str, Any] | None,
    platform_tools: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    """Return availability, route type, and a reader-facing reason."""
    if adapter_id is None:
        return (
            "unavailable",
            "unbound",
            "No adapter binding is configured for this choice.",
        )
    if adapter is None:
        return (
            "unavailable",
            "unbound",
            f"The configuration names adapter {adapter_id}, and supplies no adapter contract for it.",
        )
    native_provider = _native_provider(adapter)
    if native_provider is not None:
        status = provider_statuses.get(native_provider)
        display = {
            "modal": "Modal",
            "runpod": "RunPod",
            "lambda": "Lambda Cloud",
        }[native_provider]
        if status == provider_authorization.AUTHORIZED:
            return (
                "ready",
                "native-provider",
                f"The supplied read-only authorization report records {display} as authorized.",
            )
        if status == provider_authorization.REFUSED:
            return (
                "unavailable",
                "native-provider",
                f"The read-only authorization probe refused {display}.",
            )
        return (
            "unknown",
            "native-provider",
            f"{display} is selected, but no authorized read-only account result was supplied.",
        )
    if _has_fixture_route(adapter):
        return (
            "ready",
            "local-fixture",
            "The deterministic local fixture adapter supplies this contract route.",
        )

    platform = _platform_status(platform_tools, adapter)
    if platform == "available":
        return (
            "ready",
            "platform",
            "The supplied platform inventory records this tool as available.",
        )
    if platform == "unavailable":
        return (
            "unavailable",
            "platform",
            "The supplied platform inventory records this tool as unavailable.",
        )
    if platform == "unknown":
        return (
            "unknown",
            "platform",
            "The platform inventory contains no availability record for this tool.",
        )

    endpoint_fields = [
        provider_authorization_field
        for provider_authorization_field in provider_authorization_fields(adapter)
    ]
    if endpoint_fields:
        endpoints = _mapping(config.get("provider_endpoints"))
        invalid = [field for field in endpoint_fields if not _valid_endpoint(endpoints.get(field))]
        if invalid:
            return (
                "unavailable",
                "provider",
                f"A valid provider endpoint is missing for {', '.join(invalid)}.",
            )
        refused = [field for field in endpoint_fields if provider_statuses.get(field) == "refused"]
        if refused:
            return (
                "unavailable",
                "provider",
                f"The free authorization probe refused {', '.join(refused)}.",
            )
        unknown = [field for field in endpoint_fields if provider_statuses.get(field) != "authorized"]
        if unknown:
            return (
                "unknown",
                "provider",
                f"A valid provider endpoint is configured for {', '.join(unknown)}, and no authorized probe result was supplied.",
            )
        return (
            "ready",
            "provider",
            "The supplied free authorization report records every required provider endpoint as authorized.",
        )

    runtime = _runtime_status(runtime_checks, adapter_id)
    if runtime == "pass":
        return (
            "ready",
            "local-runtime",
            "The supplied free runtime check passed for this adapter.",
        )
    if runtime == "fail":
        return (
            "unavailable",
            "local-runtime",
            "The supplied free runtime check failed for this adapter.",
        )
    if _is_stdlib_route(adapter):
        return (
            "ready",
            "local",
            "This adapter declares a standard-library Python route.",
        )
    return (
        "unknown",
        "unmeasured",
        "This adapter declares no local route, provider endpoint, or platform binding that the chooser can verify.",
    )


def provider_authorization_fields(adapter: Mapping[str, Any]) -> tuple[str, ...]:
    """Return configured fal endpoint fields that one adapter command references."""
    token_to_field = {
        "rfdiffusion3_fal_url": "rfdiffusion3_fal_url",
        "proteinmpnn_fal_url": "proteinmpnn_fal_url",
        "esmfold2_fast_fal_url": "esmfold2_fast_fal_url",
        "alphafold_multimer_v3_fal_url": "alphafold_multimer_v3_fal_url",
    }
    return tuple(sorted(token_to_field[token] for token in _adapter_tokens(adapter) if token in token_to_field))


def _choice(
    *,
    category: str,
    identifier: str,
    adapter_id: str | None,
    adapter: Mapping[str, Any] | None,
    config: Mapping[str, Any],
    provider_statuses: Mapping[str, str],
    runtime_checks: Mapping[str, Any] | None,
    platform_tools: Mapping[str, Any] | None,
) -> dict[str, Any]:
    availability, route, reason = _availability(
        adapter_id,
        adapter,
        config,
        provider_statuses=provider_statuses,
        runtime_checks=runtime_checks,
        platform_tools=platform_tools,
    )
    return {
        "category": category,
        "id": identifier,
        "name": _display_name(identifier),
        "adapter_id": adapter_id,
        "availability": availability,
        "route": route,
        "reason": reason,
    }


def _filter_adapter_id(
    filter_id: str,
    contracts: list[Mapping[str, Any]],
    stage_adapters: Mapping[str, str],
) -> str | None:
    for contract in contracts:
        if contract.get("filter_id") != filter_id:
            continue
        stage_id = contract.get("stage_id")
        if isinstance(stage_id, str):
            return stage_adapters.get(stage_id)
    stage_id = _FILTER_STAGE_DEFAULTS.get(filter_id)
    return stage_adapters.get(stage_id) if stage_id else None


def _configured_renderers(config: Mapping[str, Any], adapters: Mapping[str, Mapping[str, Any]]) -> list[str]:
    selected: list[str] = []
    for stage in _mapping_list(config.get("stages")):
        adapter_id = stage.get("adapter_id")
        if not isinstance(adapter_id, str):
            continue
        adapter = adapters.get(adapter_id)
        if _mapping(adapter).get("role") == "renderer":
            selected.append(adapter_id)
    renderer_selection = _mapping(config.get("renderer_selection"))
    selected_id = renderer_selection.get("adapter_id")
    if isinstance(selected_id, str):
        selected.append(selected_id)
    return list(dict.fromkeys(selected))


def _campaign_choices(config: Mapping[str, Any]) -> dict[str, Any]:
    targets = _mapping_list(config.get("targets"))
    target_ids = [
        target_id
        for target in targets
        if isinstance((target_id := target.get("target_id")), str) and target_id
    ]
    sites = {
        target_id: _mapping(target.get("site")).get("mode")
        for target in targets
        if isinstance((target_id := target.get("target_id")), str) and target_id
    }
    generation = _enabled_items(config, "generation", "generators")
    counts = [item.get("backbone_count") for item in generation]
    generation_count = sum(count for count in counts if isinstance(count, int) and not isinstance(count, bool))
    return {
        "target_ids": target_ids,
        "site_modes": sites,
        "generation_count": generation_count,
        "sequence_tools": [
            _display_name(str(item["id"]))
            for item in _enabled_items(config, "sequence_design", "designers")
            if isinstance(item.get("id"), str)
        ],
        "filters": [
            _display_name(str(value))
            for value in _mapping(config.get("filters")).get("required_checks", [])
            if isinstance(value, str)
        ],
    }


def resolve_tool_menu(
    config: Mapping[str, Any],
    *,
    provider_report: Mapping[str, Any] | None = None,
    runtime_checks: Mapping[str, Any] | None = None,
    platform_tools: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return each configured tool choice with static availability evidence.

    ``provider_report`` is output that a caller already collected through the
    queue-status probe. ``runtime_checks`` and ``platform_tools`` follow the
    same pattern. This function reads those values only.
    """
    adapters = _adapter_map(config)
    stages = _stage_adapter_map(config)
    provider_statuses = _provider_statuses(provider_report)
    choices: list[dict[str, Any]] = []

    for category, section, key in (
        ("generation", "generation", "generators"),
        ("sequence design", "sequence_design", "designers"),
        ("co-folding", "cofold", "predictors"),
    ):
        for item in _enabled_items(config, section, key):
            identifier = item.get("id")
            adapter_id = item.get("adapter_id")
            if not isinstance(identifier, str) or not identifier:
                continue
            choices.append(
                _choice(
                    category=category,
                    identifier=identifier,
                    adapter_id=adapter_id if isinstance(adapter_id, str) else None,
                    adapter=adapters.get(adapter_id) if isinstance(adapter_id, str) else None,
                    config=config,
                    provider_statuses=provider_statuses,
                    runtime_checks=runtime_checks,
                    platform_tools=platform_tools,
                )
            )

    filters = _mapping(config.get("filters"))
    contracts = _mapping_list(filters.get("contracts"))
    required_checks = filters.get("required_checks")
    if not isinstance(required_checks, Sequence) or isinstance(required_checks, (str, bytes)):
        required_checks = ()
    for filter_id in required_checks:
        if not isinstance(filter_id, str):
            continue
        adapter_id = _filter_adapter_id(filter_id, contracts, stages)
        choices.append(
            _choice(
                category="filter",
                identifier=filter_id,
                adapter_id=adapter_id,
                adapter=adapters.get(adapter_id) if adapter_id else None,
                config=config,
                provider_statuses=provider_statuses,
                runtime_checks=runtime_checks,
                platform_tools=platform_tools,
            )
        )

    for adapter_id in _configured_renderers(config, adapters):
        choices.append(
            _choice(
                category="renderer",
                identifier=adapter_id,
                adapter_id=adapter_id,
                adapter=adapters.get(adapter_id),
                config=config,
                provider_statuses=provider_statuses,
                runtime_checks=runtime_checks,
                platform_tools=platform_tools,
            )
        )

    ready = [item for item in choices if item["availability"] == "ready"]
    return {
        "schema_version": MENU_SCHEMA_VERSION,
        "tools": choices,
        "ready_tools": ready,
        "campaign_choices": _campaign_choices(config),
    }


def _baseline_fidelity(config: Mapping[str, Any]) -> bool:
    """Use the executor's declared baseline-fidelity rule without copying it."""
    from . import lane

    return lane._baseline_fidelity(config)


def _selection_errors(menu: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for choice in _mapping_list(menu.get("tools")):
        if choice.get("availability") == "ready":
            continue
        name = choice.get("name", choice.get("id", "Configured tool"))
        reason = choice.get("reason", "No availability evidence was supplied.")
        errors.append(f"{name} is not ready for execution. {reason}")
    return errors


def _filter_selection_errors(config: Mapping[str, Any]) -> list[str]:
    required = _mapping(config.get("filters")).get("required_checks")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
        return []
    return [
        _RETIRED_FILTERS[filter_id]
        for filter_id in required
        if isinstance(filter_id, str) and filter_id in _RETIRED_FILTERS
    ]


def _lineage_messages(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    predictors = _enabled_items(config, "cofold", "predictors")
    predictor_ids = [
        str(item["id"])
        for item in predictors
        if isinstance(item.get("id"), str)
    ]
    if not predictor_ids:
        return [], []
    lineages = arms.enabled_predictor_lineages(config)
    if len(lineages) > 1:
        return [], []
    names = ", ".join(_display_name(identifier) for identifier in predictor_ids)
    message = f"{names} provide one predictor lineage."
    profile = _mapping(config.get("profile"))
    baseline_requested = _baseline_fidelity(config) or profile.get("baseline_fidelity") is True
    if baseline_requested:
        return [f"Baseline scoring needs an independent predictor lineage. {message}"], []
    return [], [f"Candidate scoring uses a reduced ensemble. {message}"]


def _license_errors(config: Mapping[str, Any]) -> list[str]:
    """Return failing packaged licence-gate messages without changing configuration."""
    try:
        catalog = json.loads(package_file("data", "catalog.json").read_text(encoding="utf-8"))
        report = gate.evaluate(dict(config), catalog)
    except (OSError, ValueError, json.JSONDecodeError, gate.GateError) as exc:
        return [f"The licence gate could not read the packaged catalog. {exc}"]
    return [
        str(problem["message"])
        for problem in _mapping_list(report.get("problems"))
        if problem.get("severity") == "error" and isinstance(problem.get("message"), str)
    ]


def validate_tool_selection(
    config: Mapping[str, Any],
    *,
    provider_report: Mapping[str, Any] | None = None,
    runtime_checks: Mapping[str, Any] | None = None,
    platform_tools: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Refuse execution when a selected route or claim lacks required evidence.

    Discovery and planning use ``catalog_tool_options``. They do not call this
    execution validator and do not require a package adapter.
    """
    menu = resolve_tool_menu(
        config,
        provider_report=provider_report,
        runtime_checks=runtime_checks,
        platform_tools=platform_tools,
    )
    lineage_errors, lineage_warnings = _lineage_messages(config)
    errors = _selection_errors(menu) + _filter_selection_errors(config) + lineage_errors + _license_errors(config)
    return {
        "ok": not errors,
        "errors": list(dict.fromkeys(errors)),
        "warnings": lineage_warnings,
        "menu": menu,
    }


def default_tool_selection() -> dict[str, Any]:
    """Return the packaged local-contract selection that runs without a provider."""
    campaign_path = package_file("data", "templates", "fixtures", "local-contract", "campaign.json")
    profile_path = package_file("data", "templates", "profiles", "local-contract-test.json")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    target = _mapping_list(campaign.get("targets"))[0]
    site = _mapping(target.get("site"))
    return {
        "campaign_id": campaign["campaign_id"],
        "profile_id": "local-contract-test",
        "campaign_path": str(campaign_path),
        "profile_path": str(profile_path),
        "target_id": target["target_id"],
        "site_mode": site["mode"],
        "generation_count": 3,
        "sequence_tools": ["SolubleMPNN", "ProteinMPNN"],
        "filters": [
            "Composition",
            "Exact duplicates",
            "Liability chemistry",
            "Sequence novelty",
            "Model likelihood",
        ],
        "renderer": "Browser viewer",
        "route": "local-fixture",
        "reason": "The local-contract profile uses deterministic fixture adapters and no provider-facing stage.",
    }


def _option_counts(options: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Tally the three grouping keys every catalogue option carries.

    ``screen_tool_options`` drops rows after ``catalog_tool_options`` has already
    counted them, so both callers tally the same way rather than each keeping its
    own loop over the same three keys.
    """
    role_counts: dict[str, int] = {}
    binding_counts: dict[str, int] = {}
    catalog_status_counts: dict[str, int] = {}
    for option in options:
        role = str(option.get("role"))
        binding = str(option.get("package_binding_status"))
        catalog_status = str(option.get("catalog_status"))
        role_counts[role] = role_counts.get(role, 0) + 1
        binding_counts[binding] = binding_counts.get(binding, 0) + 1
        catalog_status_counts[catalog_status] = catalog_status_counts.get(catalog_status, 0) + 1
    return {
        "roles": dict(sorted(role_counts.items())),
        "package_binding_statuses": dict(sorted(binding_counts.items())),
        "catalog_statuses": dict(sorted(catalog_status_counts.items())),
    }


def catalog_tool_options(
    *,
    catalog: Mapping[str, Any] | None = None,
    roles: Sequence[str] = (),
    statuses: Sequence[str] = (),
    live_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every recorded tool option without treating selection as execution.

    This is a static planning view. It reads the packaged catalogue and starts
    no toolcheck, account probe, endpoint request, or subprocess. A caller uses
    ``resolve_tool_menu`` after choosing a concrete profile and route.
    """
    if catalog is None:
        catalog = json.loads(package_file("data", "catalog.json").read_text(encoding="utf-8"))
    tools = catalog.get("tools")
    if not isinstance(tools, Mapping):
        raise ValueError("tool catalog must contain a tools object")

    role_filter = {str(value) for value in roles if str(value)}
    status_filter = {str(value) for value in statuses if str(value)}
    options: list[dict[str, Any]] = []
    for tool_id, raw_entry in sorted(tools.items()):
        if not isinstance(tool_id, str) or not isinstance(raw_entry, Mapping):
            continue
        availability = _mapping(raw_entry.get("availability"))
        selection = _mapping(raw_entry.get("profile_selection"))
        hardware = _mapping(raw_entry.get("hardware"))
        network = _mapping(raw_entry.get("runtime_network"))
        version = _mapping(raw_entry.get("version_or_pin"))
        role = raw_entry.get("stage_category")
        status = availability.get("status")
        if not isinstance(role, str) or not isinstance(status, str):
            continue
        if role_filter and role not in role_filter:
            continue
        adapter_id = selection.get("adapter_id")
        package_binding_status = (
            "refused"
            if status == "refused"
            else "bound"
            if isinstance(adapter_id, str) and adapter_id != "__REQUIRED__"
            else "unbound"
        )
        if status_filter and not ({status, package_binding_status} & status_filter):
            continue
        platform_tool = availability.get("platform_tool")
        unresolved = raw_entry.get("unresolved")
        options.append(
            {
                "id": tool_id,
                "name": str(raw_entry.get("display_name") or _display_name(tool_id)),
                "role": role,
                "package_binding_status": package_binding_status,
                "catalog_status": status,
                "adapter_id": adapter_id if isinstance(adapter_id, str) and adapter_id != "__REQUIRED__" else None,
                "platform_skill": (
                    platform_tool
                    if isinstance(platform_tool, str)
                    else platform_inventory.PLATFORM_SKILL_HINTS.get(tool_id)
                ),
                "live_discovery": platform_inventory.discovery_for_tool(
                    tool_id,
                    inventory=live_inventory,
                    platform_skill=platform_tool if isinstance(platform_tool, str) else None,
                ),
                "gpu_required": hardware.get("gpu_required"),
                "recommended_gpu": hardware.get("recommended_gpu"),
                "version_or_pin": version.get("value"),
                # A tool with a hosted route and an offline route records one
                # answer per route class, so hosts can be a mapping. Reading only
                # the flat list reported no host at all for those tools.
                "runtime_network_hosts": _named_hosts(network.get("hosts")),
                "runtime_network_hosts_by_route": (
                    {
                        route: _named_hosts(named)
                        for route, named in sorted(network.get("hosts", {}).items())
                    }
                    if isinstance(network.get("hosts"), Mapping)
                    else None
                ),
                "open_item_count": len(unresolved) if isinstance(unresolved, list) else 0,
            }
        )

    counts = _option_counts(options)
    return {
        "schema_version": CATALOG_MENU_SCHEMA_VERSION,
        "source": "packaged-catalog",
        "planning_only": True,
        "tool_count": len(options),
        "roles": counts["roles"],
        "package_binding_statuses": counts["package_binding_statuses"],
        "catalog_statuses": counts["catalog_statuses"],
        "live_inventory_supplied": live_inventory is not None,
        "workflow_routes": [
            {
                **workflow,
                "profiles": dict(workflow["profiles"]),
                "rules": list(workflow["rules"]),
            }
            for workflow in _WORKFLOW_ROUTES
        ],
        "tools": options,
    }


def format_catalog_tool_options(menu: Mapping[str, Any]) -> str:
    """Render the static catalogue as a compact reader-facing table."""
    rows = ["role | tool | package binding | catalog detail | live discovery | platform skill"]
    rows.append("--- | --- | --- | --- | --- | ---")
    for option in _mapping_list(menu.get("tools")):
        rows.append(
            " | ".join(
                (
                    str(option.get("role", "unknown")),
                    str(option.get("name", option.get("id", "unknown"))),
                    str(option.get("package_binding_status", "unknown")),
                    str(option.get("catalog_status", "unknown")),
                    str(_mapping(option.get("live_discovery")).get("status", "unknown")),
                    str(option.get("platform_skill") or "none recorded"),
                )
            )
        )
    rows.append("")
    rows.append(
        "Package binding describes Binder code. Live discovery describes caller-supplied platform records. Neither alone proves runtime or scientific readiness."
    )
    for workflow in _mapping_list(menu.get("workflow_routes")):
        profiles = _mapping(workflow.get("profiles"))
        rows.extend(
            (
                "",
                f"workflow: {workflow.get('id', 'unnamed')}",
                f"use when: {workflow.get('use_when', 'not recorded')}",
                "profiles: "
                + ", ".join(
                    f"{provider}={profile}"
                    for provider, profile in sorted(profiles.items())
                ),
                f"read: {workflow.get('reference', 'not recorded')}",
            )
        )
    return "\n".join(rows)


SCREEN_SCHEMA_VERSION = 1

#: The three selection sections a composed configuration names tools in. The
#: licence screen builds a one-tool configuration in the section the catalogue row
#: itself declares, so a reader of the record can see exactly what was evaluated.
#: ``gate.evaluate`` reaches its verdict from the tool id and the catalogue row,
#: not from which section holds the id, so a row whose declared section is prose
#: rather than one of these three is evaluated in the fallback section without
#: changing its verdict.
_SCREEN_SECTIONS = {
    "generation.generators": ("generation", "generators"),
    "sequence_design.designers": ("sequence_design", "designers"),
    "cofold.predictors": ("cofold", "predictors"),
}
_SCREEN_FALLBACK_SECTION = "generation.generators"

#: Every catalogue field the free-today screen reads, in the order it reads them.
#: A field that carries a ``TODO`` string, a placeholder, or nothing at all is
#: reported as unreadable and never coerced. ``hardware.gpu_required`` is the live
#: case: it is a boolean on 30 of the 31 shipped rows and a ``TODO`` sentence on
#: ncbi-sequence-fetch, so the screen compares it to ``True`` and ``False`` by
#: identity rather than testing it for truth.
FREE_TODAY_FIELDS = (
    "route_matrix.route_classes",
    "route_matrix.runnable_route_blocked_by",
    "availability.status",
    "hardware.gpu_required",
    "weights.status",
)
#: The one availability status that states the package can reach a tool with
#: nothing supplied first. The other three recorded statuses each name something
#: the operator has to provide: an install, an endpoint, a release pin, or a
#: corpus. catalog.json's availability notes carry which one per row.
AVAILABILITY_DEPLOYED = "deployed"
#: The one weights status that states a tool has no model-weight layer, so there
#: is no checkpoint to fetch before a first run. The catalogue sizes exactly one
#: weight artifact, at ``weights.bytes``, so the screen cannot tell a small
#: download from a large one and does not pretend to: any weights layer at all
#: blocks the free answer.
WEIGHTS_NOT_APPLICABLE = "not_applicable"

#: The gate refusals that come from a licence somebody has read. Two say the
#: terms forbid the use, and the third says the terms offer a separately executed
#: agreement. Every other refusal means the record itself is unsettled: a value
#: nobody filled in, a provenance reading nobody finished, or a spelling the gate
#: does not recognize.
#:
#: The list names the settled side on purpose. A refusal code this tuple does not
#: carry reports the record as unsettled, so a code the gate adds later reads as
#: "nobody has read this yet" rather than silently as "read and settled". That is
#: the conservative direction, and it is the only direction in which a stale copy
#: of the gate's vocabulary cannot overstate what is known.
SETTLED_LICENCE_REFUSALS = (
    "COMMERCIAL_USE_FORBIDDEN",
    "WEIGHTS_COMMERCIAL_USE_FORBIDDEN",
    "WRITTEN_AGREEMENT_REQUIRED",
)


def _screen_probe_section(entry: Mapping[str, Any]) -> tuple[str, tuple[str, str]]:
    """Return the selection section the licence screen evaluates one row in."""
    section = _mapping(entry.get("profile_selection")).get("section")
    if isinstance(section, str):
        for declared, pair in _SCREEN_SECTIONS.items():
            if section.startswith(declared):
                return declared, pair
    return _SCREEN_FALLBACK_SECTION, _SCREEN_SECTIONS[_SCREEN_FALLBACK_SECTION]


def commercial_use_screen(
    tool_id: str, entry: Mapping[str, Any], catalog: Mapping[str, Any]
) -> dict[str, Any]:
    """Ask the shipped licence gate what commercial use of one tool costs.

    The gate is the only thing in this package that decides a licence question,
    so the screen hands it a one-tool configuration and reports its answer. It
    reimplements no rule: a second reading of ``commercial_use`` here would drift
    from ``gate.evaluate`` the moment either side changed.

    The record carries the gate's own problem codes and catalogue citations and
    drops its prose. A gate message tells a scientist which key to edit in a
    composed configuration, and this screen runs before one exists.
    """
    section_label, (parent, key) = _screen_probe_section(entry)
    probe = {
        "declared_use": "commercial",
        parent: {key: [{"id": tool_id, "enabled": True}]},
    }
    report = gate.evaluate(dict(probe), dict(catalog), declared_use="commercial")
    problems = [
        {"code": problem["code"], "source": problem["source"]}
        for problem in report.get("problems", [])
    ]
    warning_codes = [warning["code"] for warning in report.get("warnings", [])]
    conditions = gate.stated_conditions(entry)
    conditional = bool(
        {"LICENCE_CONDITION", "WEIGHTS_COMMERCIAL_USE_CONDITIONAL"} & set(warning_codes)
    )
    if not report.get("ok"):
        verdict = "refused"
    elif conditional:
        verdict = "cleared_with_conditions"
    else:
        verdict = "cleared"
    codes = sorted({problem["code"] for problem in problems})
    # Two refusals a scientist can act on differently. A licence that forbids the
    # use rules the tool out. A licence nobody has read yet rules it out only
    # until somebody reads it, which is work a scientist can commission.
    settled = all(code in SETTLED_LICENCE_REFUSALS for code in codes)
    return {
        "verdict": verdict,
        "gate": report.get("gate"),
        "gate_ok": bool(report.get("ok")),
        "licence_record_settled": settled,
        "declared_use": "commercial",
        "evaluated_as": f"{section_label}[id={tool_id}]",
        "refusal_codes": codes,
        "problems": problems,
        "conditions": conditions,
    }


def free_today_screen(
    entry: Mapping[str, Any], route_row: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Report whether the shipped files state a tool runs today at no cost.

    Free today means all five recorded facts in ``FREE_TODAY_FIELDS`` line up: a
    shipped adapter binds the tool on a route that stays on the scientist's own
    machine, that route is not blocked by a local install the platform does not
    provide, the catalogue records the tool as deployed rather than waiting on an
    operator value, the hardware row records no GPU, and the weights row records
    no weight layer to fetch.

    The catalogue carries no price field, so this is not a priced answer. It is
    the narrower answer the recorded facts support: nothing to buy, nothing to
    provision, and nothing large to download before a first run.

    A fact that is definitely against the tool produces ``no`` and names itself
    in ``blockers``. A fact that no shipped file settles produces ``unknown`` and
    names itself in ``unreadable_fields``. A definite blocker outranks an
    unreadable field, because one settled refusal already answers the question.
    """
    from . import route_matrix

    local_classes = {route_matrix.LOCAL_PROCESS, route_matrix.LOCAL_FIXTURE}
    availability = _mapping(entry.get("availability"))
    hardware = _mapping(entry.get("hardware"))
    weights = _mapping(entry.get("weights"))

    route_classes = (route_row or {}).get("route_classes")
    blocked_by = (route_row or {}).get("runnable_route_blocked_by")
    availability_status = availability.get("status")
    gpu_required = hardware.get("gpu_required")
    weights_status = weights.get("status")

    blockers: list[str] = []
    unreadable: list[str] = []

    if route_row is None:
        unreadable.append("route_matrix.route_classes")
        local_routes: list[str] = []
    else:
        local_routes = sorted(set(route_classes or []) & local_classes)
        if not local_routes:
            blockers.append("no local route, so a first run needs a provider account")
    if isinstance(blocked_by, str) and blocked_by:
        blockers.append(f"local route blocked by {blocked_by}")
    if not isinstance(availability_status, str) or not availability_status:
        unreadable.append("availability.status")
    elif availability_status != AVAILABILITY_DEPLOYED:
        blockers.append(f"availability is {availability_status}, not deployed")
    # Identity comparison, not a truth test. ncbi-sequence-fetch records a TODO
    # sentence here, and a truth test would read that sentence as "a GPU is
    # required" and a missing key as "no GPU is required".
    if gpu_required is True:
        blockers.append("a GPU is required")
    elif gpu_required is not False:
        unreadable.append("hardware.gpu_required")
    if not isinstance(weights_status, str) or not weights_status:
        unreadable.append("weights.status")
    elif weights_status != WEIGHTS_NOT_APPLICABLE:
        blockers.append(f"weights to fetch ({weights_status}), size not recorded")

    if blockers:
        verdict = "no"
    elif unreadable:
        verdict = "unknown"
    else:
        verdict = "yes"
    return {
        "verdict": verdict,
        "definition": (
            "a shipped adapter binds a local route, that route is not blocked, "
            "availability is deployed, no GPU is required, and no model weights "
            "have to be fetched"
        ),
        "local_route_classes": local_routes,
        "blockers": blockers,
        "unreadable_fields": unreadable,
        "facts": {
            "route_matrix.route_classes": sorted(route_classes or []),
            "route_matrix.runnable_route_blocked_by": blocked_by,
            "availability.status": availability_status,
            "hardware.gpu_required": gpu_required,
            "weights.status": weights_status,
        },
    }


def screen_tool_options(
    *,
    catalog: Mapping[str, Any] | None = None,
    roles: Sequence[str] = (),
    statuses: Sequence[str] = (),
    live_inventory: Mapping[str, Any] | None = None,
    commercial_cleared_only: bool = False,
    free_today_only: bool = False,
) -> dict[str, Any]:
    """Screen the catalogue for commercial use and for what runs today at no cost.

    Two questions gate a campaign before a scientist has composed anything:
    which tools may I use commercially, and which can I run today for free. Both
    answers come from the packaged catalogue and the shipped profile templates
    alone. Nothing here needs a composed configuration, which is what
    ``gate --config`` and ``budget_plan`` both require and why neither can answer
    a question asked this early.

    Like ``catalog_tool_options``, this starts no toolcheck, account probe,
    endpoint request, or subprocess.
    """
    from . import route_matrix

    if catalog is None:
        catalog = json.loads(package_file("data", "catalog.json").read_text(encoding="utf-8"))
    tools = catalog.get("tools")
    if not isinstance(tools, Mapping):
        raise ValueError("tool catalog must contain a tools object")

    menu = catalog_tool_options(
        catalog=catalog,
        roles=roles,
        statuses=statuses,
        live_inventory=live_inventory,
    )
    route_report = route_matrix.report(live_inventory=dict(live_inventory) if live_inventory else None)
    route_rows = {
        row["tool_id"]: row
        for row in route_report.get("tools", [])
        if isinstance(row, Mapping) and isinstance(row.get("tool_id"), str)
    }

    options = [dict(option) for option in _mapping_list(menu.get("tools"))]
    for option in options:
        tool_id = str(option.get("id"))
        entry = _mapping(tools.get(tool_id))
        option["commercial_use"] = commercial_use_screen(tool_id, entry, catalog)
        option["free_today"] = free_today_screen(entry, route_rows.get(tool_id))

    commercial_counts: dict[str, int] = {}
    free_counts: dict[str, int] = {}
    unsettled: list[str] = []
    for option in options:
        commercial = option["commercial_use"]["verdict"]
        free = option["free_today"]["verdict"]
        commercial_counts[commercial] = commercial_counts.get(commercial, 0) + 1
        free_counts[free] = free_counts.get(free, 0) + 1
        if not option["commercial_use"]["licence_record_settled"]:
            unsettled.append(str(option["id"]))
    screened_count = len(options)

    if commercial_cleared_only:
        options = [
            option
            for option in options
            if option["commercial_use"]["verdict"].startswith("cleared")
        ]
    if free_today_only:
        options = [option for option in options if option["free_today"]["verdict"] == "yes"]

    counts = _option_counts(options)
    menu.update(
        {
            "schema_version": SCREEN_SCHEMA_VERSION,
            "screen_type": "claude_binder_catalog_screen",
            "source": "packaged-catalog-and-shipped-profiles",
            "licence_gate": "claude-binder.tool-licence-gate.v0",
            "meaning": (
                "Commercial use is the shipped licence gate's own verdict on a one-tool "
                "commercial campaign, so an unsettled licence reads as refused and never "
                "as permitted. Free today states that a shipped adapter binds a local "
                "route, the route is not blocked, the catalogue records the tool as "
                "deployed, no GPU is required, and no model weights have to be fetched. "
                "The catalogue carries no price field, so free today is not a priced "
                "answer and neither verdict establishes qualification or a result."
            ),
            "screened_tool_count": screened_count,
            "tool_count": len(options),
            "roles": counts["roles"],
            "package_binding_statuses": counts["package_binding_statuses"],
            "catalog_statuses": counts["catalog_statuses"],
            "commercial_use_verdicts": dict(sorted(commercial_counts.items())),
            "commercial_use_records_unsettled": sorted(unsettled),
            "free_today_verdicts": dict(sorted(free_counts.items())),
            "free_today_fields": list(FREE_TODAY_FIELDS),
            "filters": {
                "commercial_cleared_only": commercial_cleared_only,
                "free_today_only": free_today_only,
            },
            "tools": options,
        }
    )
    return menu


def format_tool_screen(screen: Mapping[str, Any]) -> str:
    """Render the screen as a reader-facing table with its definitions under it."""
    rows = ["role | tool | commercial use | free today | what stands in the way"]
    rows.append("--- | --- | --- | --- | ---")
    for option in _mapping_list(screen.get("tools")):
        commercial = _mapping(option.get("commercial_use"))
        free = _mapping(option.get("free_today"))
        codes = [str(code) for code in commercial.get("refusal_codes") or []]
        verdict = str(commercial.get("verdict", "unknown"))
        commercial_cell = f"{verdict}: {', '.join(codes)}" if codes else verdict
        obstacles = [str(item) for item in free.get("blockers") or []]
        for field in free.get("unreadable_fields") or []:
            obstacles.append(f"{field} is unsettled")
        rows.append(
            " | ".join(
                (
                    str(option.get("role", "unknown")),
                    str(option.get("name", option.get("id", "unknown"))),
                    commercial_cell,
                    str(free.get("verdict", "unknown")),
                    "; ".join(obstacles) or "nothing recorded",
                )
            )
        )
    rows.extend(
        (
            "",
            f"screened {screen.get('screened_tool_count', 0)} catalogued tools; "
            f"{screen.get('tool_count', 0)} shown after filters",
            "commercial use: "
            + ", ".join(
                f"{verdict}={count}"
                for verdict, count in _mapping(screen.get("commercial_use_verdicts")).items()
            )
            + f"; licence record unsettled on {len(screen.get('commercial_use_records_unsettled') or [])}",
            "free today: "
            + ", ".join(
                f"{verdict}={count}"
                for verdict, count in _mapping(screen.get("free_today_verdicts")).items()
            ),
            "",
            "Commercial use is the verdict of the shipped licence gate "
            f"({screen.get('licence_gate', 'unknown')}) on a one-tool commercial campaign. "
            "An unsettled or unknown licence refuses, so it is never reported as permitted. "
            "cleared_with_conditions carries obligations; read the tool's gate.conditions. "
            "A refusal on an unsettled licence record means a document nobody has read "
            "yet, which reading can settle. licence_record_settled says which kind a "
            "refusal is.",
            "Free today means all five of: a shipped adapter binds a local route, the route "
            "is not blocked by a local install the platform does not provide, the catalogue "
            "records the tool as deployed, hardware.gpu_required is false, and the weights "
            "row records no weight layer to fetch. The catalogue carries no price field and "
            "sizes only one weight artifact, so any weights layer blocks the free answer. "
            "Calling a tool free on a download nobody has measured would overstate what the "
            "catalogue records.",
            "Neither verdict establishes qualification, a price, or a scientific result.",
        )
    )
    return "\n".join(rows)
