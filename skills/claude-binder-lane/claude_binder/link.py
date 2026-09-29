"""Statically link campaign declarations to the adapter code that resolves them.

``lane link`` checks declaration-resolution divergence before a dry run or an
execution. It parses selected adapter modules with ``ast``. It does not start
commands, import provider clients, or contact a network service.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import contract_audit
from .paths import package_root as installed_package_root


CONFIG_ROOTS = frozenset(
    {
        *contract_audit.CONFIG_ROOTS,
        "provider",
        "provider_endpoints",
        "rfd3_specification",
    }
)
TERMINAL_STAGE_ROLES = frozenset({"portfolio-ranker", "output-validator", "renderer"})
SPECIAL_INPUTS = frozenset({"run-bundle", "run-bundle:controls", "stage-receipts"})
EXTERNAL_ARTIFACT_PREFIXES = frozenset({"inputs", "receipts", "stages", "validation"})
POLICY_TERMS = ("approval", "budget", "cap", "gate")

# Rules that report without stopping a run. Only orphan-declaration qualifies:
# a declared key nothing resolves leaves the code on the default it already had.
# Keep this set small. A rule belongs here when following it is optional, not
# when it is merely often wrong.
WARNING_RULES = frozenset({"orphan-declaration"})
DESCRIPTIVE_KEYS = frozenset(
    {
        "description",
        "gap_notes",
        "missing",
        "note",
        "purpose",
        "reason",
        "source",
        "threshold_basis",
    }
)

# A registered read that lands on every child of a dynamic-key mapping. No scan
# emits it. Iterating a mapping is not the same as using each entry, so the scan
# keeps reporting one unnamed sibling for `.items()`, and this marker is reserved
# for a mapping whose owner validates the namespace closed and consumes every key
# it accepts. That pairing is what makes the wildcard safe. A key the owner does
# not recognise is refused by name, which is a better report than the orphan rule
# would give.
EVERY_KEY = "{*}"

# These declarations are consumed by package code outside the selected adapter
# closure. Keep each entry specific. A broad parent declaration would hide a
# dead child from the orphan check.
PACKAGE_INTERNAL_CONFIG_CONSUMERS: Mapping[str, tuple[str, str]] = {
    # The Modal platform route reads these four off the adapter environment, but
    # modal_platform.py is not any stage's argv entry module, so it never enters a
    # stage closure and the adapter scan never opens it. Each key is held in a
    # module constant and read through it.
    "adapters[].environment.CLAUDE_BINDER_EXECUTION_ROUTE": (
        "adapters/modal_platform.py",
        'ROUTE_KEY = "CLAUDE_BINDER_EXECUTION_ROUTE"',
    ),
    "adapters[].environment.CLAUDE_BINDER_MODAL_ENV": (
        "adapters/modal_platform.py",
        'name = environment.get(ENVIRONMENT_KEY)',
    ),
    "adapters[].environment.CLAUDE_BINDER_MODAL_IMAGE": (
        "adapters/modal_platform.py",
        'image = environment.get(IMAGE_KEY)',
    ),
    "adapters[].environment.CLAUDE_BINDER_MODAL_VOLUMES": (
        "adapters/modal_platform.py",
        'raw = environment.get(VOLUMES_KEY, "{}")',
    ),
    "scoring.seed_reduction": ("report.py", '"seed_reduction",'),
    "scoring.normalization": ("report.py", '"normalization",'),
    # The runtime validator reads all three from a tuple of literal names indexed
    # into a local mapping, so the adapter scan cannot recover them. Every Modal
    # profile declares them, and without these entries a Modal campaign cannot
    # link. resources.gpu_memory_gb is read by the Modal dispatcher, which ships
    # beside the package rather than inside it, so it needs the wider search.
    "adapters[].resources.cpu": (
        "adapters/runtime_validator.py",
        'for field in ("cpu", "gpu", "memory_gb"):',
    ),
    "adapters[].resources.gpu": (
        "adapters/runtime_validator.py",
        'for field in ("cpu", "gpu", "memory_gb"):',
    ),
    "adapters[].resources.memory_gb": (
        "adapters/runtime_validator.py",
        'for field in ("cpu", "gpu", "memory_gb"):',
    ),
    "adapters[].resources.gpu_memory_gb": (
        "scripts/dispatch_modal.py",
        'requested = resources.get("gpu_memory_gb")',
    ),
    # Each of these is read as item.get(name) where item is a loop variable over
    # a config collection. The scan follows the name it starts from, so the hop
    # through the loop variable erases the key.
    "stages[].provider_facing": ("lane.py", 'provider_facing = item.get("provider_facing")'),
    "adapters[].provider_facing": ("lane.py", 'provider_facing = item.get("provider_facing")'),
    "stages[].logical_filter_stage_id": (
        "fixture_adapter.py",
        'logical_stage_id = stage.get("logical_filter_stage_id", args.stage)',
    ),
    "stages[].optimization_round": ("contract_audit.py", 'value = item.get("optimization_round", 0)'),
    "stages[].evaluation_phase": ("lane.py", 'predictor_stage["evaluation_phase"] = "optimization"'),
    # The novelty filter reaches its metric source through raw_sources[filter_id],
    # so the scan loses the concrete filter name and every field under it. Only
    # sequence_novelty is registered, which leaves a declaration under any other
    # filter id still exposed to the orphan check.
    "filters.metric_sources.sequence_novelty.kind": (
        "adapters/novelty_filter.py",
        'if source.get("kind") == TARGET_CHAIN_SOURCE_KIND:',
    ),
    "filters.metric_sources.sequence_novelty.window_length": (
        "adapters/novelty_filter.py",
        'window_length = source.get("window_length")',
    ),
    "filters.metric_sources.sequence_novelty.include_positive_controls": (
        "adapters/novelty_filter.py",
        'include_controls = source.get("include_positive_controls", True)',
    ),
    # A source naming no kind and no local structural file is read back from the
    # receipt its producing stage wrote, and produced_by names that stage. Only
    # model_likelihood declares one, because sequence_novelty carries a kind and
    # returns before this line. The concrete filter id leaves produced_by under
    # any other id exposed, which matters because nothing refuses an
    # unrecognised metric_sources key.
    "filters.metric_sources.model_likelihood.produced_by": (
        "adapters/novelty_filter.py",
        'produced_by = source.get("produced_by")',
    ),
    # The licence gate walks SELECTION_SECTIONS and reads adapter_id off each
    # selected tool. Both the section name and the list name arrive as loop
    # variables, so the scan cannot tie the read back to either declaration.
    "generation.generators[].adapter_id": (
        "gate.py",
        '"adapter_id": item.get("adapter_id"),',
    ),
    "sequence_design.designers[].adapter_id": (
        "gate.py",
        '"adapter_id": item.get("adapter_id"),',
    ),
    # The promotion selector indexes the surrogate block through a module
    # constant and then reads enabled off the local result.
    "selection.structural_diversity_surrogate.enabled": (
        "adapters/promotion_selector.py",
        'if not isinstance(raw.get("enabled"), bool):',
    ),
    # Fan-out width resolution binds count_from to a local before it reads the
    # counting mode, which hides the leaf from the lane scan.
    "stages[].fanout.count_from.value": (
        "lane.py",
        'if source.get("value") == "records":',
    ),
    # Output validation iterates stage["outputs"] and reads each contract from
    # the loop variable. The same hop hides the two sequence length bounds that
    # composition writes onto a stage and artifact validation reads back.
    "stages[].outputs[].required_fields[]": (
        "lane.py",
        'list(contract.get("required_fields", [])),',
    ),
    "stages[].sequence_minimum_length": (
        "lane.py",
        'minimum_length = int(stage.get("sequence_minimum_length", 50))',
    ),
    "stages[].sequence_maximum_length": (
        "lane.py",
        'maximum_length = int(stage.get("sequence_maximum_length", 120))',
    ),
    # The cost quote copies the whole cost_basis mapping into the row a scientist
    # reads, so the scan sees a wholesale `dict(value)` and can name no child. No
    # child can be dead either: adapter.schema.json closes the namespace with
    # additionalProperties false, and every field it lists is required unless the
    # kind is unpriced. Schema-closed plus copied whole is what earns the wildcard
    # here. cost_per_design_usd sits beside it and is read by name.
    f"adapters[].qualification.cost_basis.{EVERY_KEY}": (
        "qualify.py",
        'return dict(value) if isinstance(value, Mapping) else {}',
    ),
    "adapters[].qualification.cost_per_design_usd": (
        "qualify.py",
        'configured_per_design = _cost_value(spec.get("cost_per_design_usd"))',
    ),
    # scoring.rank_weights keys are metric names, so no schema names them and the
    # scan can only see one unnamed sibling. Its validator refuses any key that
    # does not name a selected metric and requires one weight per selected metric,
    # and _ranking_weights turns every accepted pair into a runtime weight. The
    # namespace is closed and every accepted child is live, which is what earns
    # the wildcard.
    f"scoring.rank_weights.{EVERY_KEY}": (
        "lane.py",
        'f"scoring.rank_weights key {key!r} must name scoring.primary_metric "',
    ),
}


@dataclass(frozen=True)
class Location:
    """A source position that lets a caller open the declaration or resolution."""

    file: Path
    line: int

    def text(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Producer:
    """One declaration that produces a link key."""

    owner: str
    location: Location

    def as_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "location": self.location.text()}


@dataclass(frozen=True)
class Consumer:
    """One stage whose selected adapter resolves a link key."""

    stage_id: str
    location: Location

    def as_dict(self) -> dict[str, Any]:
        return {"stage_id": self.stage_id, "location": self.location.text()}


@dataclass
class ContractKey:
    """The producer and consumers of one configuration or artifact key."""

    name: str
    producers: list[Producer] = field(default_factory=list)
    consumers: list[Consumer] = field(default_factory=list)
    terminal: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.name,
            "producers": [item.as_dict() for item in self.producers],
            "consumers": [item.as_dict() for item in self.consumers],
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class Finding:
    """One static declaration-resolution disagreement."""

    rule: str
    key: str
    location: Location
    problem: str
    fix: str

    @property
    def severity(self) -> str:
        """Return whether this finding stops a run or only reports on it.

        A declaration nothing reads cannot break a run. The stage resolves the
        default the code already carries and proceeds, so the finding is worth
        printing and must not hold execution. A read with no declaration behind
        it is the other case: the stage resolves nothing and fails.

        Without this split one false positive was exactly as fatal as one real
        defect, and the linker's own consumer model produces false positives
        whenever package code reads config from outside the selected adapter
        closure. Every lane that added correctly consumed config added orphans.
        """
        return "warning" if self.rule in WARNING_RULES else "run-failing"

    def text(self) -> str:
        return (
            f"[{self.rule}] {self.key} at {self.location.text()}: {self.problem}\n"
            f"  Fix: {self.fix}"
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "key": self.key,
            "location": self.location.text(),
            "problem": self.problem,
            "fix": self.fix,
        }


@dataclass
class LinkReport:
    """The contract table and every DRD refusal discovered from it."""

    keys: dict[str, ContractKey] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing found here will stop the run."""
        return not self.run_failing

    @property
    def run_failing(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "run-failing"]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "warning"]

    def key(self, name: str) -> ContractKey:
        return self.keys.setdefault(name, ContractKey(name=name))

    def add(self, finding: Finding) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "finding_count": len(self.findings),
            "run_failing_count": len(self.run_failing),
            "warning_count": len(self.warnings),
            "findings": [item.as_dict() for item in self.findings],
            "contract_table": [self.keys[name].as_dict() for name in sorted(self.keys)],
        }

    def text(self) -> str:
        run_failing, warnings = self.run_failing, self.warnings
        if not self.findings:
            return "link: no declaration-resolution divergence"
        if run_failing:
            lines = [f"link: {len(run_failing)} refusal(s)"]
        else:
            lines = ["link: no declaration-resolution divergence that stops a run"]
        for finding in run_failing:
            lines.extend(("", finding.text()))
        if warnings:
            lines.extend(("", f"link: {len(warnings)} warning(s), which do not stop a run"))
            for finding in warnings:
                lines.extend(("", finding.text()))
        return "\n".join(lines)


def _canonical(value: str) -> str:
    return contract_audit._canonical_config_key(value)


def _location(path: Path, key: str) -> Location:
    """Find the JSON line that declares the final field in a dotted key."""
    final = key.rsplit(".", 1)[-1].replace("[]", "").replace("{}", "")
    try:
        expression = re.compile(r'^\s*"' + re.escape(final) + r'"\s*:')
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if expression.search(line):
                return Location(path, line_number)
    except OSError:
        pass
    return Location(path, 1)


def _stage_location(config_path: Path, stage_id: str) -> Location:
    try:
        expression = re.compile(r'^\s*"stage_id"\s*:\s*"' + re.escape(stage_id) + r'"')
        for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
            if expression.search(line):
                return Location(config_path, line_number)
    except OSError:
        pass
    return Location(config_path, 1)


def _artifact_location(config_path: Path, artifact_id: str) -> Location:
    try:
        expression = re.compile(r'^\s*"artifact_id"\s*:\s*"' + re.escape(artifact_id) + r'"')
        for line_number, line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
            if expression.search(line):
                return Location(config_path, line_number)
    except OSError:
        pass
    return Location(config_path, 1)


def _is_descriptive(key: str) -> bool:
    final = key.rsplit(".", 1)[-1].replace("[]", "")
    return final.startswith("_") or final in DESCRIPTIVE_KEYS or final.endswith("_note")


def _leaf_keys(value: Any, prefix: str = "") -> Iterable[str]:
    """Yield concrete configuration leaves that can be resolved at run time."""
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str) or name == "link_terminal":
                continue
            child = f"{prefix}.{name}" if prefix else name
            if _is_descriptive(child):
                continue
            yield from _leaf_keys(item, child)
        return
    if isinstance(value, list):
        if not value:
            yield _canonical(prefix + "[]")
            return
        for item in value:
            yield from _leaf_keys(item, prefix + "[]")
        return
    if prefix:
        yield _canonical(prefix)


def _link_terminals(value: Any, prefix: str = "") -> set[str]:
    """Return declaration keys that explicitly end at this configuration boundary."""
    terminals: set[str] = set()
    if isinstance(value, Mapping):
        if value.get("link_terminal") is True and prefix:
            terminals.add(_canonical(prefix))
        for name, item in value.items():
            if isinstance(name, str) and name != "link_terminal":
                child = f"{prefix}.{name}" if prefix else name
                terminals.update(_link_terminals(item, child))
    elif isinstance(value, list):
        for item in value:
            terminals.update(_link_terminals(item, prefix + "[]"))
    return terminals


def _config_consumers(
    config: Mapping[str, Any],
    *,
    package_root: Path,
    include_optional: bool = False,
) -> list[tuple[str, Consumer]]:
    """Collect the config keys this campaign's code reads.

    include_optional decides what "reads" means. The missing-producer check asks
    whether a REQUIRED read has a declaration behind it, so it must ignore an
    optional read. The orphan-declaration check asks the opposite question,
    whether any code touches a declared key at all, and there an optional read
    counts. Reading a key through config.get() is still reading it.
    """
    adapters = {
        str(adapter.get("adapter_id")): adapter
        for adapter in config.get("adapters", [])
        if isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str)
    }
    reads: list[tuple[str, str, contract_audit.ConfigRead]] = []
    for stage in config.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_id = stage.get("stage_id")
        adapter = adapters.get(str(stage.get("adapter_id")))
        if not isinstance(stage_id, str) or adapter is None:
            continue
        for read in contract_audit._config_reads_for_stage(stage, adapter, package_root):
            if not (read.required or include_optional):
                continue
            if not contract_audit._config_read_is_reachable(config, read.dotted):
                continue
            canonical = _canonical(read.dotted)
            if _config_root(canonical) not in CONFIG_ROOTS:
                continue
            reads.append((stage_id, canonical, read))

    runtime_source = package_root / "lane.py"
    for read in contract_audit.collect_config_reads("claude_binder.lane", runtime_source):
        if not (read.required or include_optional):
            continue
        if not contract_audit._config_read_is_reachable(config, read.dotted):
            continue
        canonical = _canonical(read.dotted)
        if _config_root(canonical) not in CONFIG_ROOTS:
            continue
        reads.append(("lane-runtime", canonical, read))

    if include_optional:
        for key, consumer in _package_internal_config_consumers(package_root):
            reads.append((consumer.stage_id, key, contract_audit.ConfigRead(
                module="claude_binder",
                file=consumer.location.file,
                line=consumer.location.line,
                dotted=key,
                guarded=False,
                required=False,
                source="package-internal-registry",
            )))

    # A nested read emits each AST prefix. Keep the most specific one so a
    # mapping read does not hide an unused declared child such as seed_aggregation.
    result: list[tuple[str, Consumer]] = []
    for stage_id, key, read in reads:
        if any(
            other_stage == stage_id
            and other_key != key
            and other_key.startswith(key + ".")
            for other_stage, other_key, _ in reads
        ):
            continue
        result.append((key, Consumer(stage_id, Location(read.file, read.line))))
    return result


def _package_internal_config_consumers(package_root: Path) -> list[tuple[str, Consumer]]:
    """Return declared package readers the lane and adapter scan cannot recover.

    Each registered reader indexes a configuration block into a local variable
    and then reads a field off that variable. The AST scan follows the name it
    starts from, so a variable hop or a loop variable erases the concrete key.
    The registry names those keys explicitly and verifies the marker line before
    it vouches for one, which keeps orphan detection intact for every sibling.
    """
    consumers: list[tuple[str, Consumer]] = []
    for key, (relative_path, marker) in PACKAGE_INTERNAL_CONFIG_CONSUMERS.items():
        for base in _consumer_search_roots(package_root):
            source = base / relative_path
            try:
                lines = source.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line_number, line in enumerate(lines, start=1):
                if marker in line:
                    consumers.append((key, Consumer("package-internal", Location(source, line_number))))
                    break
            else:
                continue
            break
    return consumers


def _consumer_search_roots(package_root: Path) -> tuple[Path, ...]:
    """Return the roots a registered consumer file may live under.

    The Modal dispatcher reads adapter resources but ships beside the package
    rather than inside it, and the two layouts put it in different places. A
    built skill holds it at ``<skill>/scripts``. This repository holds it at
    ``skills/claude-binder-lane/scripts``. Try both, so one registry serves both.
    """
    return (
        package_root,
        package_root.parent,
        package_root.parents[1] / "skills" / "claude-binder-lane"
        if len(package_root.parents) > 1
        else package_root,
    )


def _config_root(key: str) -> str:
    """Return the top-level configuration name a key hangs off.

    A read of a list element renders its first segment as `adapters[]`, while
    CONFIG_ROOTS names the root `adapters`. Comparing the two forms directly
    discarded every read under `targets`, `stages`, `adapters` and
    `license_gates` before it could vouch for anything, which left every leaf
    under those four roots outside the orphan check entirely.
    """
    return key.split(".", 1)[0].removesuffix("[]")


def _keys_overlap(left: str, right: str) -> bool:
    normalized_left = re.sub(r"\.?\[\]|\.?\{\}", "", left)
    normalized_right = re.sub(r"\.?\[\]|\.?\{\}", "", right)
    return (
        normalized_left == normalized_right
        or normalized_left.startswith(normalized_right + ".")
        or normalized_right.startswith(normalized_left + ".")
    )


_WILDCARD_SEGMENT = re.compile(r"^\{.*\}$")


def _key_segments(key: str) -> list[str]:
    """Split a dotted key into its named segments.

    The per-element marker rides on the key it iterates, and a list hop carries
    no name of its own, so controls.positive[].gates[].metric is four segments.
    """
    return [part for part in (raw.replace("[]", "") for raw in key.split(".")) if part]


def _key_at_or_below(consumer_key: str, key: str) -> bool:
    """True when consumer_key names key itself or something inside it.

    Reading scoring.rank_weights proves scoring.rank_weights is used. Reading
    the whole scoring mapping does not prove any particular child is, which is
    why an ancestor read must not count as consuming its descendants. Letting it
    count is what lets a dead key hide under a broad config.get("scoring", {}).

    A {name} segment is a read through a variable key. The scan cannot say which
    sibling it lands on, so it matches every sibling at that depth.
    """
    consumed = _key_segments(consumer_key)
    declared = _key_segments(key)
    if len(consumed) < len(declared):
        return False
    # A wildcard the code then reads through, as in controls[group]["id"], is a
    # container hop and lands on every sibling, so it vouches for the concrete
    # leaf named after it. A wildcard in the last position names no leaf at all,
    # so it must not vouch for any particular one, or every child of a mapping
    # the code iterates would look consumed.
    #
    # EVERY_KEY is the exception, and it never arrives from the scan. It comes
    # only from a registry entry whose marker line was checked, so a wildcard in
    # the last position is a claim someone verified rather than a guess the
    # scanner made about syntax.
    final = consumed[len(declared) - 1]
    if _WILDCARD_SEGMENT.match(final) is not None and final != EVERY_KEY:
        return False
    return all(
        left == right
        or _WILDCARD_SEGMENT.match(left) is not None
        or _WILDCARD_SEGMENT.match(right) is not None
        for left, right in zip(consumed, declared)
    )


def _build_config_table(
    report: LinkReport,
    config: Mapping[str, Any],
    config_path: Path,
    *,
    package_root: Path,
) -> None:
    # Two questions, two consumer sets. missing-producer asks whether a required
    # read has a declaration behind it, so it uses the required-only set.
    # orphan-declaration asks whether anything reads a declared key, and an
    # optional read answers that, so every entry records the wider set.
    required_consumers = _config_consumers(config, package_root=package_root)
    consumers = _config_consumers(
        config, package_root=package_root, include_optional=True
    )
    required_pairs = set(required_consumers)
    consumer_roots = {_config_root(key) for key, _ in consumers}
    terminals = _link_terminals(config)

    for key, consumer in consumers:
        entry = report.key(f"config.{key}")
        if consumer not in entry.consumers:
            entry.consumers.append(consumer)
        if (key, consumer) not in required_pairs:
            continue
        # The lane runtime uses optional configuration branches while it validates
        # and materializes a plan. Only a selected adapter can make a missing key a
        # link refusal. A wildcard key identifies a dynamic selector, so the linker
        # cannot name one missing declaration precisely enough to reject it.
        if (
            consumer.stage_id != "lane-runtime"
            and "{" not in key
            and "[" not in key
            and not contract_audit._config_has(config, key)
        ):
            report.add(
                Finding(
                    rule="missing-producer",
                    key=f"config.{key}",
                    location=consumer.location,
                    problem=(
                        f"stage {consumer.stage_id} resolves this required key, but no campaign or profile declaration produces it"
                    ),
                    fix=f"declare {key} in the campaign or its bound profile",
                )
            )

    for key in _leaf_keys(config):
        if _config_root(key) not in consumer_roots:
            continue
        entry = report.key(f"config.{key}")
        producer = Producer("configuration", _location(config_path, key))
        if producer not in entry.producers:
            entry.producers.append(producer)
        entry.terminal = entry.terminal or any(_keys_overlap(key, terminal) for terminal in terminals)
        for consumer_key, consumer in consumers:
            # A read only vouches for the key it actually names and what lies
            # inside it, whether the read is required or not. Reading the whole
            # scoring object proves scoring is used; it proves nothing about any
            # particular child, and counting it as proof is what lets a dead key
            # hide under a broad read of its parent.
            if _key_at_or_below(consumer_key, key) and consumer not in entry.consumers:
                entry.consumers.append(consumer)

    for name, entry in report.keys.items():
        if not name.startswith("config.") or entry.consumers or entry.terminal:
            continue
        producer = entry.producers[0]
        report.add(
            Finding(
                rule="orphan-declaration",
                key=name,
                location=producer.location,
                problem="the configuration declares this key, but no selected adapter resolves it",
                fix="remove the declaration, bind an adapter that resolves it, or set link_terminal: true on its object",
            )
        )


def _stage_artifact_key(stage_id: str, artifact_id: str) -> str:
    return f"artifact.{stage_id}:{artifact_id}"


def _published_artifact_key(relative: str, published: Mapping[str, str]) -> str | None:
    exact = published.get(relative)
    if exact is not None:
        return exact
    matches = [path for path in published if "*" in path and fnmatch(relative, path)]
    if matches:
        return published[matches[0]]
    # A read path can itself carry a wildcard, because the static scan renders a
    # loop or format variable as "*". Match that against the concrete publish
    # paths too, otherwise a per-round artifact reads as unproduced even though
    # every round publishes it.
    if "*" in relative:
        globbed = sorted(path for path in published if fnmatch(path, relative))
        if globbed:
            return published[globbed[0]]
    return None


def _add_ast_artifact_consumers(
    report: LinkReport,
    config: Mapping[str, Any],
    config_path: Path,
    published: Mapping[str, str],
    *,
    package_root: Path,
) -> None:
    adapters = {
        str(adapter.get("adapter_id")): adapter
        for adapter in config.get("adapters", [])
        if isinstance(adapter, Mapping) and isinstance(adapter.get("adapter_id"), str)
    }
    missing_paths: set[tuple[str, Location]] = set()
    optimization = config.get("optimization")
    optimization_enabled = bool(
        optimization.get("enabled") if isinstance(optimization, Mapping) else False
    )
    for stage in config.get("stages", []):
        if not isinstance(stage, Mapping) or not isinstance(stage.get("stage_id"), str):
            continue
        stage_id = stage["stage_id"]
        adapter = adapters.get(str(stage.get("adapter_id")))
        if adapter is None:
            continue
        entry_module = contract_audit._entry_module(adapter.get("command_argv_template") or [])
        if entry_module is None:
            continue
        analysis_entry = contract_audit._analysis_entry(entry_module)
        closure = contract_audit.module_closure(analysis_entry, package_root, depth=None)
        if not closure:
            continue
        reachable = contract_audit.reachable_functions(
            analysis_entry,
            closure,
            adapter.get("command_argv_template") or [],
            stage_id=stage_id,
            adapter_id=str(adapter.get("adapter_id")),
        )
        for module_name, source in closure.items():
            for use in contract_audit.collect_path_uses(
                module_name,
                source,
                reachable_functions=reachable.functions.get(module_name, frozenset()),
                reachable_nodes=reachable.nodes.get(module_name, frozenset()),
            ):
                if use.root != contract_audit.SHARED_ROOT or use.mode != "read" or not use.relative:
                    continue
                relative = use.relative.strip("/")
                first = Path(relative).parts[0] if relative else ""
                if first in EXTERNAL_ARTIFACT_PREFIXES or "<dynamic>" in relative:
                    continue
                # A round-scoped read lives behind a round-number guard, so only an
                # enabled optimization loop can reach it and only that loop publishes
                # it. Demanding a producer from a campaign that disables optimization
                # asks for a stage the composer never emits.
                if not optimization_enabled and first == "optimization":
                    continue
                location = Location(use.file, use.line)
                consumer = Consumer(stage_id, location)
                name = _published_artifact_key(relative, published)
                if name is not None:
                    entry = report.key(name)
                    if consumer not in entry.consumers:
                        entry.consumers.append(consumer)
                    continue
                name = f"artifact-path.{relative}"
                entry = report.key(name)
                if consumer not in entry.consumers:
                    entry.consumers.append(consumer)
                missing_path = (name, location)
                if missing_path in missing_paths:
                    continue
                missing_paths.add(missing_path)
                report.add(
                    Finding(
                        rule="missing-producer",
                        key=name,
                        location=location,
                        problem=(
                            f"stage {stage_id} resolves this artifact path, but no stage output publishes it"
                        ),
                        fix="declare a producing stage output with this publish_path or remove the read",
                    )
                )


def _build_artifact_table(
    report: LinkReport,
    config: Mapping[str, Any],
    config_path: Path,
    *,
    package_root: Path,
) -> None:
    stages = [stage for stage in config.get("stages", []) if isinstance(stage, Mapping)]
    output_keys: dict[str, list[str]] = {}
    published: dict[str, str] = {}
    for stage in stages:
        stage_id = stage.get("stage_id")
        if not isinstance(stage_id, str):
            continue
        for output in stage.get("outputs", []) or []:
            if not isinstance(output, Mapping) or not isinstance(output.get("artifact_id"), str):
                continue
            artifact_id = output["artifact_id"]
            name = _stage_artifact_key(stage_id, artifact_id)
            entry = report.key(name)
            producer = Producer(stage_id, _artifact_location(config_path, artifact_id))
            if producer not in entry.producers:
                entry.producers.append(producer)
            entry.terminal = entry.terminal or stage.get("required_role") in TERMINAL_STAGE_ROLES
            output_keys.setdefault(stage_id, []).append(name)
            publish_path = output.get("publish_path")
            if isinstance(publish_path, str) and publish_path:
                published.setdefault(publish_path.strip("/"), name)

    for stage in stages:
        stage_id = stage.get("stage_id")
        if not isinstance(stage_id, str):
            continue
        location = _stage_location(config_path, stage_id)
        references = list(stage.get("inputs", []) or [])
        fanout = stage.get("fanout")
        if isinstance(fanout, Mapping) and isinstance(fanout.get("count_from"), Mapping):
            count_from = fanout["count_from"]
            if isinstance(count_from.get("stage_id"), str) and isinstance(count_from.get("artifact_id"), str):
                references.append(f"{count_from['stage_id']}:{count_from['artifact_id']}")
        for reference in references:
            if not isinstance(reference, str) or reference in SPECIAL_INPUTS or ":" not in reference:
                continue
            source_stage, artifact_id = reference.split(":", 1)
            name = _stage_artifact_key(source_stage, artifact_id)
            entry = report.key(name)
            consumer = Consumer(stage_id, location)
            if consumer not in entry.consumers:
                entry.consumers.append(consumer)
            if not entry.producers:
                report.add(
                    Finding(
                        rule="missing-producer",
                        key=name,
                        location=location,
                        problem=f"stage {stage_id} consumes this artifact, but no stage declares it as an output",
                        fix=f"add {source_stage}:{artifact_id} to a producing stage or remove the input",
                    )
                )

        # A dependency can consume several declared sidecar outputs. It gives each
        # output a stage consumer even when the adapter receives only derived argv.
        for source_stage in stage.get("depends_on", []) or []:
            if not isinstance(source_stage, str):
                continue
            for name in output_keys.get(source_stage, []):
                entry = report.key(name)
                consumer = Consumer(stage_id, location)
                if consumer not in entry.consumers:
                    entry.consumers.append(consumer)

    _add_ast_artifact_consumers(
        report,
        config,
        config_path,
        published,
        package_root=package_root,
    )

    for name, entry in report.keys.items():
        if not name.startswith("artifact.") or entry.consumers or entry.terminal:
            continue
        producer = entry.producers[0]
        report.add(
            Finding(
                rule="orphan-declaration",
                key=name,
                location=producer.location,
                problem="this stage output has no downstream consumer and is not terminal",
                fix="add a dependent stage, mark the output terminal, or remove the output declaration",
            )
        )


def _roster_path(config: Mapping[str, Any], config_path: Path, package_root: Path) -> Path | None:
    runtime = config.get("runtime")
    configured = runtime.get("model_roster_path") if isinstance(runtime, Mapping) else None
    if not isinstance(configured, str) or not configured:
        return None
    candidate = Path(configured)
    choices = [
        candidate if candidate.is_absolute() else config_path.parent / candidate,
        package_root / candidate,
        package_root / "data" / candidate.name,
    ]
    return next((path.resolve() for path in choices if path.is_file()), None)


def _structure_hashes(package_root: Path) -> set[str]:
    hashes: set[str] = set()
    for suffix in ("*.pdb", "*.cif", "*.mmcif"):
        for path in (package_root / "data").rglob(suffix):
            if path.is_file():
                hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes


def _campaign_target_hashes(config: Mapping[str, Any], config_path: Path) -> set[str]:
    """Hash the target structures this campaign itself declares.

    The roster inventory rule asks whether the bytes behind a qualified target
    hash can be found. Scanning only the package's own ``data`` directory answers
    that for the shipped roster, whose rows deliberately record an unshipped
    hash under a different field name and so never reach the rule. A roster
    written by ``lane qualify`` against an operator's own target records a real
    ``target_structure_sha256``, and those bytes live in the campaign and the
    materialized bundle rather than inside the installed package. Reading them
    here is what makes the rule answerable for that roster instead of always
    false.
    """
    hashes: set[str] = set()
    targets = config.get("targets")
    if not isinstance(targets, list):
        return hashes
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        for field in ("structure_path", "runtime_structure_path", "structure_source_path"):
            value = target.get(field)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = config_path.parent / path
            if path.is_file():
                hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes


def _check_roster_inventory(
    report: LinkReport,
    config: Mapping[str, Any],
    config_path: Path,
    *,
    package_root: Path,
) -> None:
    roster_path = _roster_path(config, config_path, package_root)
    if roster_path is None:
        return
    try:
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    rows = roster.get("models", []) if isinstance(roster, Mapping) else []
    if not isinstance(rows, list):
        return
    inventory = _structure_hashes(package_root) | _campaign_target_hashes(config, config_path)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        digest = row.get("target_structure_sha256")
        if not isinstance(digest, str) or digest in inventory:
            continue
        location = _location(roster_path, "target_structure_sha256")
        report.add(
            Finding(
                rule="missing-structure-inventory",
                key="target_structure_sha256",
                location=location,
                problem="the roster declares a target structure hash that matches no shipped structure",
                fix="ship the qualified structure bytes or mark the roster row as unshipped and non-authorizing",
            )
        )


def _policy_values(value: Any, prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not isinstance(value, Mapping):
        return values
    for key, item in value.items():
        if not isinstance(key, str) or key in {"policy_waivers", "link_policy_waivers"}:
            continue
        child = f"{prefix}.{key}" if prefix else key
        if any(term in key.lower() for term in POLICY_TERMS):
            values[child] = item
        elif isinstance(item, Mapping):
            values.update(_policy_values(item, child))
    return values


def _policy_waivers(stage: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for name in ("policy_waivers", "link_policy_waivers"):
        values = stage.get(name)
        if isinstance(values, list):
            result.update(value for value in values if isinstance(value, str))
    return result


def _check_policy_parity(report: LinkReport, config: Mapping[str, Any], config_path: Path) -> None:
    by_role: dict[str, list[Mapping[str, Any]]] = {}
    for stage in config.get("stages", []):
        if isinstance(stage, Mapping) and isinstance(stage.get("required_role"), str):
            by_role.setdefault(stage["required_role"], []).append(stage)
    for role, siblings in by_role.items():
        if len(siblings) < 2:
            continue
        policies = {str(stage.get("stage_id")): _policy_values(stage) for stage in siblings}
        all_keys = set().union(*(set(values) for values in policies.values()))
        for key in sorted(all_keys):
            for stage in siblings:
                stage_id = str(stage.get("stage_id"))
                if key in policies[stage_id]:
                    continue
                waivers = _policy_waivers(stage)
                if "all" in waivers or key in waivers or key.rsplit(".", 1)[-1] in waivers:
                    continue
                location = _stage_location(config_path, stage_id)
                report.add(
                    Finding(
                        rule="policy-parity",
                        key=f"policy.{role}.{key}",
                        location=location,
                        problem=(
                            f"sibling stage {stage_id} omits a gate, cap, budget, or approval declared by another {role} path"
                        ),
                        fix=f"declare {key} on {stage_id} or add it to policy_waivers",
                    )
                )


def _check_ranking_arm_contradiction(
    report: LinkReport,
    config: Mapping[str, Any],
    config_path: Path,
) -> None:
    scoring = config.get("scoring")
    cofold = config.get("cofold")
    if not isinstance(scoring, Mapping) or not isinstance(cofold, Mapping):
        return
    declared = scoring.get("ranking_arms", scoring.get("ranking_instruments"))
    predictors = cofold.get("predictors")
    if not isinstance(declared, list) or not isinstance(predictors, list):
        return
    enabled = [
        item
        for item in predictors
        if isinstance(item, Mapping) and item.get("enabled", True) is True
    ]
    if not enabled or len(declared) <= len(enabled):
        return
    field = "ranking_arms" if "ranking_arms" in scoring else "ranking_instruments"
    location = _location(config_path, f"scoring.{field}")
    report.add(
        Finding(
            rule="contradictory-declarations",
            key=f"config.scoring.{field}",
            location=location,
            problem=(
                f"the ranking declaration names {len(declared)} arms, but the bound cofold profile enables {len(enabled)} predictor path(s)"
            ),
            fix="align the ranking arms with enabled predictors or declare an explicit single-arm waiver",
        )
    )


def link_config(
    config: Mapping[str, Any],
    config_path: Path,
    *,
    package_root: Path | None = None,
) -> LinkReport:
    """Link one composed campaign without executing a campaign command."""
    root = (package_root or installed_package_root()).resolve()
    source = Path(config_path).resolve()
    report = LinkReport()
    _build_config_table(report, config, source, package_root=root)
    _build_artifact_table(report, config, source, package_root=root)
    _check_roster_inventory(report, config, source, package_root=root)
    _check_policy_parity(report, config, source)
    _check_ranking_arm_contradiction(report, config, source)
    report.findings.sort(key=lambda item: (item.rule, item.key, item.location.text()))
    return report


def link_path(config_path: Path, *, package_root: Path | None = None) -> LinkReport:
    """Load a composed campaign JSON document and link it."""
    source = Path(config_path)
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError(f"composed campaign must be a JSON object: {source}")
    return link_config(config, source, package_root=package_root)
