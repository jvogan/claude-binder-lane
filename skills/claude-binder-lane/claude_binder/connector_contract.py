"""Inspect a resolved adapter contract without importing or executing its tools.

The report describes package configuration. Live endpoint compatibility and
scientific qualification require separate evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import lane
from .paths import package_file


def required_fields(value: Any, path: str = "") -> list[str]:
    """Locate unresolved declarations, including nested profile overrides."""
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in required_fields(child, f"{path}.{key}" if path else key)
        ]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in required_fields(child, f"{path}[{index}]")
        ]
    return [path] if lane.is_required_placeholder(value) else []


def command_modules(adapter: dict[str, Any]) -> dict[str, str | None]:
    """Read Python module names as data; never import a selected adapter."""
    result = {}
    for field in ("toolcheck_argv", "command_argv_template", "parser_argv_template"):
        argv = adapter.get(field, [])
        module = None
        if isinstance(argv, list) and "-m" in argv:
            index = argv.index("-m") + 1
            if index < len(argv) and isinstance(argv[index], str):
                module = argv[index]
        result[field] = module
    return result


def inspect_contract(profile_path: Path, adapter_id: str | None = None) -> dict[str, Any]:
    """Return inherited declarations, not an authorization or readiness verdict."""
    profile = lane.load_profile(profile_path)
    adapters = profile.get("adapters", [])
    if not isinstance(adapters, list) or any(not isinstance(a, dict) for a in adapters):
        raise ValueError("profile.adapters must be a list of objects")
    ids = [a.get("adapter_id") for a in adapters]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("profile adapters must have unique non-empty adapter_id values")
    if adapter_id is not None and adapter_id not in ids:
        raise ValueError(f"unknown adapter {adapter_id!r}; available: {', '.join(ids)}")
    selected = [a for a in adapters if adapter_id is None or a["adapter_id"] == adapter_id]
    contracts = []
    for adapter in selected:
        stages = [
            stage for stage in profile.get("stages", [])
            if stage.get("adapter_id") == adapter["adapter_id"]
        ]
        item = {
            "adapter_id": adapter["adapter_id"],
            "role": adapter.get("role"),
            "modules": command_modules(adapter),
            "stage_ids": [stage["stage_id"] for stage in stages],
            "required_adapter_fields": required_fields(adapter),
        }
        if adapter_id is not None:
            item["adapter"] = adapter
            item["stages"] = stages
            item["required_stage_fields"] = required_fields(stages, "stages")
        contracts.append(item)
    return {
        "schema_version": 1,
        "inspection": "resolved-profile-contract",
        "provider_calls": 0,
        "readiness": "not_assessed",
        "scope": "Adapter and stage declarations; campaign inputs and live route evidence are separate.",
        "authoring_reference": "references/connector-authoring.md",
        "contracts": contracts,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Packaged profile filename or a local profile/config path")
    parser.add_argument("--adapter", help="Include this adapter's full contract and attached stages; omit to list adapters")
    args = parser.parse_args(argv)
    try:
        supplied = Path(args.profile).expanduser()
        if supplied.is_file():
            profile_path = supplied
        elif supplied.name == args.profile:
            profile_path = package_file("data", "templates", "profiles", args.profile)
        else:
            raise FileNotFoundError(f"profile does not exist: {args.profile}")
        report = inspect_contract(profile_path, args.adapter)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"inspection": "failed", "error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
