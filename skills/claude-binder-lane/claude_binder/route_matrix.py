#!/usr/bin/env python3
"""Report which execution routes each catalogued tool already has.

Decision 3 and decision 4 ask a scientist to choose a compute route and a tool
stack together. The packaged catalogue answers capability, licence, weights,
hardware and adapter binding per tool. It does not answer the route question,
so that answer has lived in prose and in twenty-six profile templates.

This module derives it from shipped files. For every tool the catalogue records,
it reports each profile template that enables the tool, the route class that
profile runs it on, and whether the named adapter module exists. A live Claude
Science inventory is optional and stays separate: a registered endpoint or a
visible platform skill is reported as discovery, never as a package route.

Every route value is read from the tree. The vocabulary is not: the route class
names, the provider lifecycle map, the role suffixes and the section list are
literals in this file, and ``platform_inventory.PLATFORM_SKILL_HINTS`` is a
hand-kept table. Those decide how a fact is named. They do not supply one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from . import lane, platform_inventory
from .paths import package_root
from .reproduction_readiness import _module_name
from .tool_menu import _has_fixture_route

SCHEMA_VERSION = 1

# A closed route vocabulary. Each value states what the shipped files prove and
# nothing more. None of them establishes qualification, price, or a licence.
LOCAL_PROCESS = "local-process"
LOCAL_FIXTURE = "local-fixture"
HOSTED_DEPLOYMENT = "hosted-deployment"
SELF_HOSTED = "self-hosted-provider"
HOSTED_API = "hosted-api"
#: The platform owns the model service. A campaign registers the endpoint once
#: through the `managed-model-endpoints` skill and then calls it by its
#: registered name, so the profile carries no URL, port or credential. Nothing
#: in this package binds one yet, which is why no adapter can produce this class
#: and `platform_managed_endpoints` reports it as an unbound surface instead.
PLATFORM_MANAGED_ENDPOINT = "platform-managed-endpoint"

FAL_URL_RE = re.compile(r"\{\{[a-z0-9_]*fal_url\}\}")
ROLE_SUFFIXES = ("_generator", "_designer", "_predictor", "_renderer", "_scorer", "_filter")
PLATFORM_MODULES = {
    "modal": "modal_platform",
    "runpod": "runpod_platform",
    "lambda": "lambda_platform",
}
SECTIONS = (
    ("generation", "generators"),
    ("sequence_design", "designers"),
    ("cofold", "predictors"),
)


def profiles_dir() -> Path:
    return package_root() / "data" / "templates" / "profiles"


def adapters_dir() -> Path:
    return package_root() / "adapters"


def _argv_text(adapter: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("command_argv_template", "toolcheck_argv", "parser_argv_template"):
        parts.extend(str(token) for token in adapter.get(key) or [])
    return " ".join(parts)


def _route_class(profile: dict[str, Any], adapter: dict[str, Any] | None, module: str | None) -> str:
    """Classify one adapter row in one profile by what that profile declares."""
    # Order matters. A profile that names a provider can still send one arm to a
    # hosted application, so what the adapter itself declares is read before the
    # profile-level provider. Provider first labelled every row on a Modal
    # profile self-hosted, including its fal arms.
    #
    # The fixture is recognised from the argv rather than from `module`, and it
    # has to be. `_module_name` matches `claude_binder.adapters.<name>` because
    # readiness uses its answer to decide whether `adapters/<name>.py` exists,
    # while the fixture ships at the package root and runs as
    # `claude_binder.fixture_adapter`. Broadening that matcher would send
    # readiness looking for an adapters file that is not there. `tool_menu`
    # already reads the same command to report a local-fixture route, so this
    # calls that predicate rather than writing a second one. Testing `module`
    # here labelled no row at all, and every fixture-bound row read
    # local-process.
    if adapter is not None and _has_fixture_route(adapter):
        return LOCAL_FIXTURE
    if adapter is not None and (
        FAL_URL_RE.search(_argv_text(adapter)) or (module or "").startswith("fal_")
    ):
        return HOSTED_DEPLOYMENT
    provider_id = ((profile.get("provider") or {}).get("provider_id")) if isinstance(
        profile.get("provider"), dict
    ) else None
    if isinstance(provider_id, str) and provider_id:
        return SELF_HOSTED
    return LOCAL_PROCESS


def _provider_id(profile: dict[str, Any]) -> str | None:
    provider = profile.get("provider")
    if not isinstance(provider, dict):
        return None
    value = provider.get("provider_id")
    return value if isinstance(value, str) and value else None


def profile_routes(profile: dict[str, Any], profile_name: str) -> list[dict[str, Any]]:
    """Return one row per enabled tool in one resolved profile."""
    adapters = {
        row["adapter_id"]: row
        for row in profile.get("adapters") or []
        if isinstance(row, dict) and isinstance(row.get("adapter_id"), str)
    }
    rows: list[dict[str, Any]] = []
    for section, key in SECTIONS:
        block = profile.get(section)
        if not isinstance(block, dict):
            continue
        for item in block.get(key) or []:
            if not isinstance(item, dict):
                continue
            tool_id = item.get("id")
            adapter_id = item.get("adapter_id")
            if not isinstance(tool_id, str):
                continue
            adapter = adapters.get(adapter_id) if isinstance(adapter_id, str) else None
            module = _module_name(adapter) if adapter else None
            module_present = bool(module) and (adapters_dir() / f"{module}.py").is_file()
            rows.append(
                {
                    "tool_id": tool_id,
                    "role": key,
                    "profile": profile_name,
                    "adapter_id": adapter_id if isinstance(adapter_id, str) else None,
                    "adapter_module": module,
                    "adapter_module_present": module_present,
                    "route_class": _route_class(profile, adapter, module),
                    "provider_id": _provider_id(profile),
                    "enabled": item.get("enabled") is True,
                }
            )
    return rows


def _catalog_adapter_ids(entry: dict[str, Any]) -> list[str]:
    """Return the adapter ids the catalogue names for a tool selected outside the stack.

    A catalogue row can name more than one adapter, as ``novelty-filter / msa-builder``
    does, and can name a script rather than an adapter id. Keep the clean ids only.
    """
    raw = (entry.get("profile_selection") or {}).get("adapter_id")
    if not isinstance(raw, str) or raw == "__REQUIRED__":
        return []
    ids = []
    for part in raw.split("/"):
        candidate = part.strip()
        if candidate and " " not in candidate and not candidate.endswith(".py"):
            ids.append(candidate)
    return ids


def _stack_selectable(entry: dict[str, Any]) -> bool:
    section = (entry.get("profile_selection") or {}).get("section")
    if not isinstance(section, str):
        return False
    return any(section.startswith(f"{block}.{key}") for block, key in SECTIONS)


def adapter_row_routes(
    profile: dict[str, Any], profile_name: str, adapter_ids: set[str]
) -> list[dict[str, Any]]:
    """Return one row per named adapter this profile carries outside the stack sections."""
    rows: list[dict[str, Any]] = []
    for adapter in profile.get("adapters") or []:
        if not isinstance(adapter, dict):
            continue
        adapter_id = adapter.get("adapter_id")
        if not isinstance(adapter_id, str) or adapter_id not in adapter_ids:
            continue
        module = _module_name(adapter)
        rows.append(
            {
                "role": adapter.get("role"),
                "profile": profile_name,
                "adapter_id": adapter_id,
                "adapter_module": module,
                "adapter_module_present": bool(module)
                and (adapters_dir() / f"{module}.py").is_file(),
                "route_class": _route_class(profile, adapter, module),
                "provider_id": _provider_id(profile),
                "enabled": None,
            }
        )
    return rows


def platform_adapter_use() -> dict[str, dict[str, Any]]:
    """Report which provider lifecycle modules ship and which profiles select one."""
    # Two separate facts. A profile declares a provider in provider.provider_id,
    # which is what a campaign selects. A profile may additionally carry a params
    # block named after the lifecycle module. Searching the raw text for the
    # module name found only the second, so two profiles that declare Modal went
    # unreported.
    declared: dict[str, list[str]] = {}
    for path in sorted(profiles_dir().glob("*.json")):
        try:
            resolved = lane.load_profile(path)
        except Exception:  # an unresolvable template is reported by report()
            continue
        provider = _provider_id(resolved)
        if provider:
            declared.setdefault(provider, []).append(path.name)
    text = {path.name: path.read_text(encoding="utf-8") for path in sorted(profiles_dir().glob("*.json"))}
    report: dict[str, dict[str, Any]] = {}
    for provider, module in PLATFORM_MODULES.items():
        report[provider] = {
            "lifecycle_module": module,
            "lifecycle_module_present": (adapters_dir() / f"{module}.py").is_file(),
            "profiles": sorted(declared.get(provider, [])),
            "profiles_naming_the_lifecycle_module": [
                name for name, body in text.items() if module in body
            ],
        }
    return report


#: The two platform skills that make up the managed model endpoint family, with
#: the file each contract was read from. Both are byte-identical from
#: `0.1.41-release` through the installed `0.1.47-release`.
PLATFORM_ENDPOINT_SKILLS = ("managed-model-endpoints", "using-model-endpoint")


def platform_managed_endpoints() -> dict[str, Any]:
    """Report the platform's hosted-model surface and whether this package binds it.

    Modal, RunPod and Lambda Cloud each get a row in ``platform_adapters`` because
    each has a lifecycle module in this package. The managed endpoint family has
    none, and it never will in that shape, because the platform's inference
    provider refuses ``create_sandbox``, ``exec``, ``list_owned``, ``read_owner``
    and ``terminate``. There is no job to submit, attach to, or settle. Reporting
    the family here rather than leaving it out keeps a reader from concluding that
    a route the platform ships does not exist.
    """
    bound = sorted(
        {
            row["adapter_id"]
            for path in sorted(profiles_dir().glob("*.json"))
            for row in _managed_endpoint_rows(path)
            if row.get("adapter_id")
        }
    )
    return {
        "platform_skills": list(PLATFORM_ENDPOINT_SKILLS),
        "contract": "skills/claude-binder-lane/references/managed-endpoint-route.md",
        "has_job_lifecycle": False,
        "credential_name": "NVIDIA_API_KEY",
        "adapters_bound": bound,
        "meaning": (
            "The platform registers a model service once and then reaches it by name. "
            "A campaign can use it by hand today. No adapter in this package binds one, "
            "so no catalogued tool carries the platform-managed-endpoint route class. "
            "The route has no submit, resume, receipt or settlement step, so an adapter "
            "for it has to carry its own evidence rather than reuse the provider lifecycle."
        ),
    }


def _managed_endpoint_rows(path: Path) -> list[dict[str, Any]]:
    """Return adapter rows that select the managed endpoint route in one profile."""
    try:
        resolved = lane.load_profile(path)
    except Exception:  # an unresolvable template is reported by report()
        return []
    rows = []
    for adapter in resolved.get("adapters") or []:
        if not isinstance(adapter, dict):
            continue
        environment = adapter.get("environment")
        if isinstance(environment, dict) and environment.get(
            "CLAUDE_BINDER_EXECUTION_ROUTE"
        ) == "managed-endpoint":
            rows.append(adapter)
    return rows


_DECLARED_TOOL_ID_RE = re.compile(r"^([A-Z][A-Z0-9_]*_ID) = \"([^\"]+)\"", re.MULTILINE)
_TOOL_ID_CONSTANT_WORDS = ("TOOL", "PREDICTOR", "GENERATOR", "DESIGNER", "RENDERER", "SCORER", "FILTER")


def _module_owners(modules: set[str], tool_ids: set[str]) -> dict[str, str]:
    """Attribute each shipped adapter module to one catalogue row.

    Two rules, both read from the tree. A module that declares its own tool id in
    a module-level constant is attributed by that id. Otherwise the longest tool
    prefix that matches the module name wins, with a leading ``fal_`` stripped
    first. Plain ``startswith`` put ``boltz2_local_predictor`` on ``boltz``
    instead of ``boltz-local``, and the ``fal_`` prefix hid every hosted module
    from every row, so ``fal_genie3_generator`` was reported under no tool at all.
    """
    prefixes = sorted(
        ((tool_id.replace("-", "_"), tool_id) for tool_id in tool_ids),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    owners: dict[str, str] = {}
    for module in modules:
        if not module.endswith(ROLE_SUFFIXES):
            continue
        source = (adapters_dir() / f"{module}.py").read_text(encoding="utf-8")
        declared = next(
            (
                value
                for name, value in _DECLARED_TOOL_ID_RE.findall(source)
                if name.rsplit("_", 1)[0].endswith(_TOOL_ID_CONSTANT_WORDS) and value in tool_ids
            ),
            None,
        )
        if declared:
            owners[module] = declared
            continue
        stem = module[4:] if module.startswith("fal_") else module
        for prefix, tool_id in prefixes:
            if stem.startswith(prefix):
                owners[module] = tool_id
                break
    return owners


def _accelerated_build_summary(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Summarise the published accelerated kit for one tool, or None.

    The catalogue gained `accelerated_build` blocks for the tools the 2026-09-17
    release covers, and for a week nothing read them. A grep for the key across
    every Python file in this package returned only the catalogue itself, so the
    pins, the modes and the measured speed-ups were reference prose stored in
    JSON. This lifts them into the one report that answers what a tool can do,
    beside the route that would run it.

    It states native setup separately from graph dispatch. `dispatchable_here`
    describes a full graph binding; it is echoed alongside shipped recipes and
    native qualification evidence rather than treated as a capability limit.
    """
    block = entry.get("accelerated_build")
    if not isinstance(block, dict):
        return None
    speedups = block.get("forward_pass_speedup")
    fastest = None
    if isinstance(speedups, dict):
        measured = [
            (mode, value)
            for mode, value in speedups.items()
            if isinstance(value, (int, float))
        ]
        if measured:
            fastest = max(measured, key=lambda item: item[1])
    return {
        "kit": block.get("kit"),
        "modes": block.get("modes"),
        "fastest_published_mode": fastest[0] if fastest else None,
        "fastest_published_speedup": fastest[1] if fastest else None,
        "dispatchable_here": block.get("dispatchable_here"),
        "environment_recipe": block.get("modal_environment_recipe"),
        # Output formats alone do not establish requested acceleration.
        # Read activation, manifests and runtime counters together.
        "mode_verifier": block.get("mode_verifier"),
        # Native kit setup is usable independently of a full Binder graph binding.
        "native_setup_reference": block.get("native_setup_reference"),
        "native_execution": block.get("native_execution"),
    }


def report(*, live_inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    catalog = json.loads((package_root() / "data" / "catalog.json").read_text(encoding="utf-8"))
    tools = catalog.get("tools")
    if not isinstance(tools, dict):
        raise ValueError("tool catalog must contain a tools object")

    # A tool the stack sections do not select is reached through a named adapter
    # row instead. Scoring, novelty search, and rendering tools are that shape,
    # and reading only the stack sections reported them as having no route.
    # One adapter row can serve two catalogued tools. ``interface-scorer``
    # computes both ipSAE and DockQ, and ``viewer-renderer`` covers PyMOL and
    # ChimeraX, so an owner map that kept one tool per adapter reported the
    # other as having no route.
    adapter_owners: dict[str, list[str]] = {}
    for tool_id, entry in tools.items():
        if not isinstance(entry, dict) or _stack_selectable(entry):
            continue
        for adapter_id in _catalog_adapter_ids(entry):
            adapter_owners.setdefault(adapter_id, []).append(tool_id)

    rows: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for path in sorted(profiles_dir().glob("*.json")):
        try:
            resolved = lane.load_profile(path)
        except Exception as error:  # a template that cannot resolve is a fact, not a crash
            unreadable.append({"profile": path.name, "error": type(error).__name__})
            continue
        rows.extend(profile_routes(resolved, path.name))
        for row in adapter_row_routes(resolved, path.name, set(adapter_owners)):
            for owner in adapter_owners[row["adapter_id"]]:
                rows.append({**row, "tool_id": owner})

    # A row the profile does not mark hosted can still leave the machine. Boltz
    # binds the hosted Boltz Cloud API and was reported as local-process, which
    # is the answer a scientist choosing a compute route would act on. The
    # catalogue states the job-time fact per tool, so use it rather than infer
    # from the module name.
    # A row whose adapter module is absent proves nothing about job-time network,
    # so a contract-test row keeps the class the profile declared.
    for row in rows:
        if row["route_class"] != LOCAL_PROCESS or not row["adapter_module_present"]:
            continue
        network = (tools.get(row["tool_id"]) or {}).get("runtime_network") or {}
        offline = network.get("offline_capable_at_job_time")
        named_hosts = network.get("hosts")
        if offline is False and isinstance(named_hosts, list) and named_hosts:
            row["route_class"] = HOSTED_API

    by_tool: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_tool.setdefault(row["tool_id"], []).append(row)

    # A module can ship in the adapters directory with no profile naming it. The
    # code is then present and unreachable, which is a different answer from
    # absent and matters to a scientist deciding whether to swap a tool in.
    bound_modules = {row["adapter_module"] for row in rows if row["adapter_module"]}
    shipped_modules = {path.stem for path in adapters_dir().glob("*.py")}
    module_owner = _module_owners(shipped_modules, set(tools))

    entries: list[dict[str, Any]] = []
    for tool_id in sorted(tools):
        entry = tools[tool_id]
        if not isinstance(entry, dict):
            continue
        tool_rows = by_tool.get(tool_id, [])
        unbound_modules = sorted(
            module
            for module, owner in module_owner.items()
            if owner == tool_id and module not in bound_modules
        )
        runnable = [row for row in tool_rows if row["adapter_module_present"]]
        # A shared adapter row gives a tool a route that belongs to its neighbour.
        # viewer-renderer serves PyMOL and ChimeraX, and the module behind it
        # requires a PyMOL executable, so counting ChimeraX as routed overstated
        # the answer. The catalogue already records the condition, so read it
        # rather than restate it here.
        # This holds an availability status, not a route fact, and the name
        # oversells it. It is set only for `conditional_local_install`, which is
        # chimerax and pymol-open-source, where the catalogue records a local
        # install the platform does not provide. Every other tool reads null here,
        # including the tools that have no route class at all, so a null does not
        # mean a route exists and is unblocked.
        blocked = (
            (entry.get("availability") or {}).get("status")
            if (entry.get("availability") or {}).get("status") == "conditional_local_install"
            else None
        )
        entries.append(
            {
                "tool_id": tool_id,
                "name": entry.get("display_name") or tool_id,
                "role": entry.get("stage_category"),
                "catalog_status": (entry.get("availability") or {}).get("status"),
                "selection_kind": (
                    "stack-row"
                    if _stack_selectable(entry)
                    else "adapter-row"
                    if _catalog_adapter_ids(entry)
                    else "no-profile-selector"
                ),
                "route_classes": sorted({row["route_class"] for row in runnable}),
                "route_classes_contract_only": sorted(
                    {row["route_class"] for row in tool_rows if not row["adapter_module_present"]}
                ),
                "providers": sorted({row["provider_id"] for row in runnable if row["provider_id"]}),
                "shipped_modules_no_profile_binds": unbound_modules,
                "runnable_route_blocked_by": blocked,
                "profile_routes": sorted(
                    tool_rows, key=lambda row: (row["profile"], row["adapter_id"] or "")
                ),
                "live_discovery": platform_inventory.discovery_for_tool(
                    tool_id,
                    inventory=live_inventory,
                    platform_skill=(entry.get("availability") or {}).get("platform_tool"),
                ),
                "accelerated_build": _accelerated_build_summary(entry),
            }
        )

    class_counts: dict[str, int] = {}
    for item in entries:
        for route_class in item["route_classes"]:
            class_counts[route_class] = class_counts.get(route_class, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "check_type": "claude_binder_route_matrix",
        "planning_only": True,
        "meaning": (
            "A route class states that a shipped profile binds the tool that way. "
            "It does not establish qualification, price, licence clearance, or a scientific result. "
            "A tool whose runnable_route_blocked_by is set has route rows through an adapter it "
            "shares with another tool, and the catalogue records a local install the platform "
            "does not provide, so it is not counted as having a runnable route. "
            "A tool with no route class has no shipped profile binding here. That states what a "
            "Binder run plan can dispatch, not what a scientist can run: the catalogue names a "
            "Claude Science platform skill for some of these tools in availability.platform_tool, "
            "and a scientist runs those by hand today. Read live_discovery.skill_name per tool, "
            "and note that live_discovery.status stays unknown until a platform inventory is "
            "supplied, because this package cannot confirm a skill is installed without one. "
            "A measurement taken on a hosted-api or hosted-deployment route does not transfer to "
            "self-hosting the same software: it exercises the vendor's build on the vendor's "
            "hardware, so it prices that route and says nothing about another one."
        ),
        "catalog_tool_count": len(entries),
        "profiles_read": len({row["profile"] for row in rows}),
        "profiles_unreadable": unreadable,
        "tools_with_a_runnable_route": sum(
            1
            for item in entries
            if item["route_classes"] and not item["runnable_route_blocked_by"]
        ),
        # A tool can have no profile binding here and still be something a
        # scientist runs today through the platform. Counting the two facts
        # separately stops a reader taking "no route" for "cannot be run".
        # A published accelerated kit exists for some of these tools, and an
        # environment recipe ships for fewer. A recipe does not establish a
        # full graph binding, so `dispatchable_here` is echoed per tool rather
        # than summarised into a single reassuring number.
        "tools_with_an_accelerated_kit": sum(
            1 for item in entries if item["accelerated_build"]
        ),
        "tools_with_an_accelerated_environment_recipe": sorted(
            item["tool_id"]
            for item in entries
            if (item["accelerated_build"] or {}).get("environment_recipe")
        ),
        "tools_with_a_platform_skill_named": sum(
            1 for item in entries if item["live_discovery"]["skill_name"]
        ),
        "tools_with_no_route_but_a_platform_skill": sorted(
            item["tool_id"]
            for item in entries
            if not item["route_classes"] and item["live_discovery"]["skill_name"]
        ),
        "route_class_tool_counts": dict(sorted(class_counts.items())),
        "platform_adapters": platform_adapter_use(),
        "platform_managed_endpoints": platform_managed_endpoints(),
        "live_inventory_supplied": live_inventory is not None,
        "tools": entries,
    }


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Tool | Role | Route classes with a shipped module | Providers | Contract-only routes | Live discovery |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in result["tools"]:
        lines.append(
            "| {tool} | {role} | {classes} | {providers} | {contract} | {live} |".format(
                tool=item["tool_id"],
                role=item["role"] or "-",
                classes=", ".join(item["route_classes"]) or "none",
                providers=", ".join(item["providers"]) or "-",
                contract=", ".join(item["route_classes_contract_only"]) or "-",
                live=(
                    "{status} (platform skill {skill})".format(
                        status=item["live_discovery"]["status"],
                        skill=item["live_discovery"]["skill_name"],
                    )
                    if item["live_discovery"]["skill_name"]
                    else item["live_discovery"]["status"]
                ),
            )
        )
    kits = [item for item in result["tools"] if (item["accelerated_build"] or {}).get("environment_recipe")]
    if kits:
        lines += ["", "| Accelerated tool | Native setup | Execution evidence | Full graph binding |",
                  "| --- | --- | --- | --- |"]
        for item in kits:
            kit = item["accelerated_build"]
            native = kit.get("native_execution") or {}
            lines.append("| {tool} | {recipe} | {status} | {bound} |".format(
                tool=item["tool_id"], recipe=kit["environment_recipe"],
                status=native.get("status", "read the selected kit's qualification record"),
                bound="yes" if kit["dispatchable_here"] else "none",
            ))
    platform = result["platform_adapters"]
    lines.append("")
    lines.append("| Provider lifecycle | Module ships | Profiles that select it |")
    lines.append("| --- | --- | --- |")
    for provider, record in sorted(platform.items()):
        lines.append(
            "| {provider} | {present} | {profiles} |".format(
                provider=provider,
                present="yes" if record["lifecycle_module_present"] else "no",
                profiles=", ".join(record["profiles"]) or "none",
            )
        )
    endpoints = result.get("platform_managed_endpoints")
    if endpoints:
        lines.append("")
        lines.append("| Platform endpoint family | Job lifecycle | Adapters in this package that bind it |")
        lines.append("| --- | --- | --- |")
        lines.append(
            "| {skills} | {lifecycle} | {bound} |".format(
                skills=", ".join(endpoints["platform_skills"]),
                lifecycle="yes" if endpoints["has_job_lifecycle"] else "none",
                bound=", ".join(endpoints["adapters_bound"]) or "none",
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform-inventory",
        help="JSON snapshot collected through Claude Science skill and compute discovery",
    )
    parser.add_argument("--tool", action="append", default=[], help="include one tool id; repeat to add more")
    parser.add_argument("--markdown", action="store_true", help="print the route table")
    args = parser.parse_args(argv)
    snapshot = None
    if args.platform_inventory:
        snapshot = json.loads(Path(args.platform_inventory).read_text(encoding="utf-8"))
    result = report(live_inventory=snapshot)
    if args.tool:
        wanted = set(args.tool)
        result["tools"] = [item for item in result["tools"] if item["tool_id"] in wanted]
    if args.markdown:
        print(markdown(result))
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
