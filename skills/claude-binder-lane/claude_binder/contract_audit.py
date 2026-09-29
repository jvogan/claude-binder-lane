#!/usr/bin/env python3
"""Compare a materialized run plan against the adapter code it will run.

Nine blockers stopped the first real executor run on 2026-08-23. A dry run had
passed all sixteen stages, validation had returned zero refusals, and the suite
was green. All nine reduce to two mechanisms, and this module checks for both.

**Undeclared read.** An adapter opens a filesystem path, or requires a config
key, that no earlier stage declares as an output and no bundled input supplies.
The adapter's contract lives in its code. The plan is maintained by hand. The
two drift, and nothing compares them.

**Contract against emission.** A stage declares ``required_fields``, a ``kind``
or a count for one of its outputs, and the adapter that emits that output
writes something else.

The second mechanism is the expensive one. Those checks run after the predictor
returns, so an unfixed instance buys GPU predictions and then discards them.

The audit reads the plan, the resolved config and the adapter source. Parser
inspection imports each selected command module and calls its parser builder.
The audit dispatches zero commands and sends zero network requests. Run it
before authorizing a paid run.

Method, and why it is this one
------------------------------
An adapter declares ``accepted_artifacts`` and ``produced_artifacts``, and
comparing those two lists against the plan is the cheap route. It is also the
route that would have caught none of the nine. Those lists name artifact
*types*, the code opens *paths*, and every one of the nine defects lived in the
gap between the two. ``esmfold2-fast-predictor`` accepts
``rescore-candidate-manifest`` and the plan's ``promote`` stage produces one, so
the declaration check passes while the reader opens a path the plan never
publishes.

So this walks the code. For each adapter bound to a stage it resolves the
entry function selected by ``command_argv_template``, follows its first-party
call graph, and evaluates command and stage branches that the argv selects.
It parses the selected entry module and its first-party imports, then folds the
reachable ``Path`` arithmetic into relative paths under a named root. It also
reads the ``{{artifact_root}}`` tokens out of the argv templates themselves,
because two of the nine were argv-only and appear nowhere in adapter code.

One check needs no code at all. A stage that requires a field on its rows, and
whose adapter never writes that field, has to get it from an earlier stage. If
the only stage in the plan that declares the field runs later, the plan
contradicts itself, and that holds whatever the adapters do next.

Recall is worth more than precision here, so an unresolvable segment is
reported with the uncertainty named rather than dropped. Every finding carries
the file and line it came from. Nothing is inferred from a name.

Limits, stated so a reader does not over-trust a pass:

- ``lane`` is excluded. It is the executor rather than an adapter, nearly every
  adapter imports it for a validation helper, and reading its paths as an
  adapter's contract produced two hundred findings about stages this plan does
  not run. A read that lives only in ``lane`` is missed.
- Import following stops one level past the entry module, and a module that
  extends ``sys.path`` before importing is not followed at all. Findings about
  such a module are downgraded and say so.
- Guard detection covers the enclosing branch, the short-circuited operand and
  the early-return prologue. A function called only from inside a branch reads
  as unguarded.
- Whether an adapter writes a field is judged by whether it names the field
  anywhere. A module that copies an upstream row wholesale can carry a field it
  never names, so its absences are smells.
- ``records_per_count`` for a structure kind cannot be derived statically. That
  one is reported as a smell that names the counting rule in force.
- Nothing here reads file formats. An adapter that opens the right path with a
  parser that cannot read that path's format passes every check.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import contextlib
import importlib
import importlib.util
import inspect
import io
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


# Severity. A run-failing finding stops the run; it is what gates a paid run.
# A smell is a real disagreement that a guard, a disabled feature or an unbound
# adapter currently hides. It costs nothing now and costs a run when the guard
# moves.
RUN_FAILING = "run-failing"
SMELL = "smell"

# Confidence. `certain` means the finding is read directly off the file named in
# `location`. `uncertain` means the analysis could not resolve part of the
# expression and the reader has to open the line.
CERTAIN = "certain"
UNCERTAIN = "uncertain"

# The names an adapter uses for a filesystem root. `artifact_root` is the only
# one that is shared between stages, so it is the only root a producer can be
# demanded for. The others are named so their reads can be recognized and set
# aside rather than silently skipped.
SHARED_ROOT = "artifact_root"
PRIVATE_ROOTS = frozenset(
    {"attempt_dir", "receipts_dir", "out_dir", "work_dir", "workspace", "tmp_dir", "round_root"}
)
ROOT_NAMES = frozenset({SHARED_ROOT, "run_root", "bundle_path", "bundle_root", *PRIVATE_ROOTS})

# Callables that read a path, and callables that write one. A path that reaches
# neither is reported as an unclassified use rather than assumed to be either.
READER_CALLS = frozenset(
    {
        "read_jsonl",
        "read_json",
        "read_json_object",
        "load_json",
        "load_jsonl",
        "read_text",
        "read_bytes",
        "open",
        "is_file",
        "exists",
        "is_dir",
        "iterdir",
        "glob",
        "rglob",
        "stat",
        "resolve_existing",
        "validate_filter_cohort",
    }
)
WRITER_CALLS = frozenset(
    {
        "write_text",
        "write_bytes",
        "write_json",
        "write_jsonl",
        "atomic_write_json",
        "atomic_write_jsonl",
        "atomic_write_bytes",
        "mkdir",
        "touch",
        "unlink",
        "rmtree",
        "copy",
        "copy2",
        "copyfile",
        "replace",
        "rename",
        "dump",
        "dump_json",
        "makedirs",
    }
)

# argv flags whose value names a file the adapter creates. Every other flag
# carrying a path is treated as a read, which is the recall-first direction: a
# misread write produces one dismissable finding, a missed read produces none.
WRITE_FLAGS = frozenset(
    {
        "--out",
        "--out-dir",
        "--output",
        "--output-dir",
        "--receipt",
        "--receipts-dir",
        "--manifest-path",
        "--result-path",
        "--sequence-out-dir",
        "--log",
        "--log-path",
    }
)

# What `lane.validate_artifact` treats as one record, per kind. A declaration
# that carries `required_fields` for a kind outside the first two has those
# fields silently ignored.
# A key path part meaning "an element of the container named just before it".
# A field read inside a loop is required only when that loop runs, so an absent
# or empty container makes every field under the marker optional. Without it a
# loop over a `metric_registry` the profile never sets reported three missing
# fields per stage that reads it.
ELEMENT_OF = "[]"

# These conditions are stated by the validator and by the adapter branches
# that use them. They let the audit evaluate a conditional read against the
# resolved campaign instead of treating every source-code subscript as live.
CONDITIONAL_SELECTION_GATE = "targets[].selection_gate"
CONDITIONAL_METRIC_SOURCES = "filters.metric_sources"

# The campaign-level contig keys sit behind two guards in generator_preflight.
# _check_generation_request returns at :826 unless the bound generator runs the
# legacy RFdiffusion adapter, and the value only comes from the campaign at :836
# when that adapter passes --contigs a {{contigs}} or {{generator_contigs}}
# template token. A profile that writes a literal value, or the __REQUIRED__
# placeholder every shipped profile writes, keeps the value in the argv and never
# reads these keys. They are the four _campaign_contigs tries at :426 and :431.
CONTIG_ARGV_OPTION = "--contigs"
# The same pattern generator_preflight.py:35 uses to decide whether an argv
# value is a template token rather than a literal contig string.
CONTIG_ARGV_TOKEN_RE = re.compile(r"^\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}$")
CONTIG_CAMPAIGN_TOKENS = frozenset({"contigs", "generator_contigs"})
CONDITIONAL_CONTIG_KEYS = frozenset(
    {
        "contigs",
        "generator_contigs",
        "generation.contigs",
        "generation.generator_contigs",
    }
)
OPTIMIZATION_ROUND_PATH_PREFIX = "optimization/rounds/round-"

FIELD_CHECKED_KINDS = frozenset({"json", "jsonl"})

# `lane` is the executor, not an adapter. Almost every adapter imports it for a
# validation helper, and it carries the paths of every stage the template can
# ever emit, including rounds this plan does not run. Reading it as part of an
# adapter's contract produced two hundred findings that belong to the
# orchestrator rather than to the stage. It is analyzed only when an argv names
# it directly, which is how the ranker runs.
LIBRARY_MODULES = frozenset({"claude_binder.lane"})
STRUCTURE_KINDS = frozenset({"pdb", "cif", "structure"})
KIND_SUFFIXES: Mapping[str, frozenset[str]] = {
    "json": frozenset({".json"}),
    "jsonl": frozenset({".jsonl"}),
    "fasta": frozenset({".fasta", ".fa", ".faa"}),
    "pdb": frozenset({".pdb"}),
    "cif": frozenset({".cif", ".mmcif"}),
    "structure": frozenset({".pdb", ".cif", ".mmcif"}),
    "image": frozenset({".png", ".jpg", ".jpeg", ".svg"}),
}


@dataclass(frozen=True)
class Finding:
    """One disagreement between the plan and the code that will run it.

    The shape follows ``generator_preflight.Problem``: a ``field`` naming what
    is wrong, a ``problem`` saying what was observed and a ``fix`` saying what
    to change. Three fields are added, because a preflight problem is read
    while looking at one campaign file and an audit finding is read while
    looking at sixteen stages: ``stage`` and ``adapter`` say where it fires,
    and ``location`` says which file and line it was read from.
    """

    stage: str
    adapter: str
    location: str
    field: str
    problem: str
    fix: str
    severity: str = RUN_FAILING
    confidence: str = CERTAIN

    def as_dict(self) -> dict[str, str]:
        return {
            "stage": self.stage,
            "adapter": self.adapter,
            "location": self.location,
            "field": self.field,
            "problem": self.problem,
            "fix": self.fix,
            "severity": self.severity,
            "confidence": self.confidence,
        }

    def text(self) -> str:
        mark = "" if self.confidence == CERTAIN else " [uncertain]"
        return (
            f"[{self.severity}]{mark} {self.stage} / {self.adapter} / {self.location}\n"
            f"  {self.field}: {self.problem}\n"
            f"  Fix: {self.fix}"
        )


@dataclass
class AuditReport:
    """Every finding, and every check that was actually performed."""

    findings: list[Finding] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def run_failing(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == RUN_FAILING]

    @property
    def smells(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == SMELL]

    @property
    def ok(self) -> bool:
        """True when nothing found would fail the run.

        A smell does not gate. It is reported so a person decides.
        """
        return not self.run_failing

    def add(self, finding: Finding) -> None:
        if finding not in self.findings:
            self.findings.append(finding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "run_failing_count": len(self.run_failing),
            "smell_count": len(self.smells),
            "checked": list(self.checked),
            "skipped": list(self.skipped),
            "findings": [item.as_dict() for item in self.findings],
        }

    def text(self) -> str:
        if not self.findings:
            return "contract audit: no findings"
        lines = [
            f"contract audit: {len(self.run_failing)} run-failing, {len(self.smells)} smell"
        ]
        for finding in [*self.run_failing, *self.smells]:
            lines.append("")
            lines.append(finding.text())
        return "\n".join(lines)


# --- path arithmetic, folded out of the source -------------------------------


@dataclass(frozen=True)
class PathUse:
    """One ``root / segment / segment`` expression found in adapter source."""

    module: str
    file: Path
    line: int
    root: str
    relative: str
    guarded: bool
    resolved: bool
    mode: str  # "read", "write" or "unclassified"

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


def _strip_calls(node: ast.AST) -> ast.AST:
    """Return the expression under any chain of ``.resolve()`` style calls."""
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"resolve", "expanduser", "absolute"}:
                node = node.func.value
                continue
        return node


def _root_name(node: ast.AST) -> str | None:
    """Return the filesystem root a path expression is anchored to."""
    node = _strip_calls(node)
    if isinstance(node, ast.Name):
        return node.id if node.id in ROOT_NAMES else None
    if isinstance(node, ast.Attribute):
        return node.attr if node.attr in ROOT_NAMES else None
    return None


def _fold_constant(node: ast.AST, constants: Mapping[str, str]) -> str | None:
    """Fold one path segment to text, or return None when it cannot be read."""
    node = _strip_calls(node)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.Attribute):
        return constants.get(node.attr)
    if isinstance(node, ast.JoinedStr):
        # An f-string names a value the audit cannot know, so the varying part
        # becomes a wildcard and the finding is marked unresolved.
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("*")
        return "".join(parts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        if len(node.args) == 1:
            return _fold_constant(node.args[0], constants)
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _fold_constant(node.left, constants)
        right = _fold_constant(node.right, constants)
        if left is None or right is None:
            return None
        return str(PurePosixPath(left) / right)
    return None


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """Return module-level names that fold to a relative path or a string."""
    constants: dict[str, str] = {}
    for statement in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign):
            targets = list(statement.targets)
            value = statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            targets = [statement.target]
            value = statement.value
        if value is None:
            continue
        folded = _fold_constant(value, constants)
        if folded is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = folded
    return constants


def _split_path(node: ast.AST, constants: Mapping[str, str]) -> tuple[str, str, bool] | None:
    """Return ``(root, relative, resolved)`` for one path expression.

    ``resolved`` is False when a segment could not be folded. The segment then
    reads as ``<dynamic>`` so a person can see which part is unknown.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
        return None
    segments: list[str] = []
    resolved = True
    current: ast.AST = node
    while isinstance(current, ast.BinOp) and isinstance(current.op, ast.Div):
        folded = _fold_constant(current.right, constants)
        if folded is None:
            folded = "<dynamic>"
            resolved = False
        elif "*" in folded:
            # An f-string segment folds to text with a wildcard where the value
            # goes. The text is worth reporting and the wildcard is the part the
            # reader has to resolve, so the finding says so.
            resolved = False
        segments.append(folded)
        current = current.left
    root = _root_name(current)
    if root is None:
        return None
    segments.reverse()
    return root, str(PurePosixPath(*segments)) if segments else "", resolved


class _GuardMap:
    """Answer whether a node is reached only under a condition.

    Three shapes count as a guard. A node inside an ``if`` body or ``else``
    body. A node in the second or later operand of an ``and``, which the first
    operand short-circuits. And a node that follows an ``if`` whose body ends in
    ``return`` or ``raise``, which is how every adapter here writes an early
    exit.

    A node inside an ``if`` test is not guarded by that test, because the test
    always evaluates.
    """

    def __init__(self, tree: ast.Module) -> None:
        self._guarded: set[int] = set()
        self._walk_block(tree.body, guarded=False)

    def _walk_block(self, body: Sequence[ast.stmt], *, guarded: bool) -> None:
        prologue_guard = guarded
        for statement in body:
            if isinstance(statement, ast.If):
                self._mark_expr(statement.test, guarded=prologue_guard, in_test=True)
                self._walk_block(statement.body, guarded=True)
                self._walk_block(statement.orelse, guarded=True)
                if _block_exits(statement.body):
                    prologue_guard = True
                continue
            if isinstance(statement, (ast.For, ast.While, ast.AsyncFor)):
                self._mark_expr(getattr(statement, "iter", None) or statement.test, guarded=prologue_guard)
                self._walk_block(statement.body, guarded=prologue_guard)
                self._walk_block(statement.orelse, guarded=True)
                continue
            if isinstance(statement, ast.Try):
                self._walk_block(statement.body, guarded=prologue_guard)
                for handler in statement.handlers:
                    self._walk_block(handler.body, guarded=True)
                self._walk_block(statement.orelse, guarded=True)
                self._walk_block(statement.finalbody, guarded=prologue_guard)
                continue
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._walk_block(statement.body, guarded=False)
                continue
            if isinstance(statement, ast.With):
                self._walk_block(statement.body, guarded=prologue_guard)
                for item in statement.items:
                    self._mark_expr(item.context_expr, guarded=prologue_guard)
                continue
            for child in ast.iter_child_nodes(statement):
                if isinstance(child, ast.expr):
                    self._mark_expr(child, guarded=prologue_guard)

    def _mark_expr(self, node: ast.expr | None, *, guarded: bool, in_test: bool = False) -> None:
        if node is None:
            return
        if guarded:
            self._guarded.add(id(node))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for index, operand in enumerate(node.values):
                self._mark_expr(operand, guarded=guarded or index > 0, in_test=in_test)
            return
        if isinstance(node, ast.IfExp):
            self._mark_expr(node.test, guarded=guarded)
            self._mark_expr(node.body, guarded=True)
            self._mark_expr(node.orelse, guarded=True)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                self._mark_expr(child, guarded=guarded, in_test=in_test)

    def is_guarded(self, node: ast.AST) -> bool:
        return id(node) in self._guarded


def _block_exits(body: Sequence[ast.stmt]) -> bool:
    """True when a block always leaves the function."""
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


_TREES: dict[Path, ast.Module | None] = {}
_MODULE_CLOSURE_CACHE: dict[tuple[str, Path, int | None], dict[str, Path]] = {}
_STAGE_CONFIG_READ_CACHE: dict[
    tuple[Path, str, str, str], tuple[ConfigRead, ...]
] = {}


def _parse(source_path: Path) -> ast.Module | None:
    """Parse one module once per process. lane.py alone is half a megabyte."""
    if source_path in _TREES:
        return _TREES[source_path]
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError):
        tree = None
    _TREES[source_path] = tree
    return tree


def _scopes(tree: ast.Module, reachable_functions: Iterable[str] | None = None) -> list[ast.AST]:
    """Return the module and selected functions as separate binding scopes.

    A variable name means different things in two functions. Binding names
    module-wide made ``source`` in one function inherit a config subtree bound
    to ``source`` in another, which invented fields nobody reads.

    ``None`` keeps the full-module behavior used by package-wide indexes. A set
    limits the scan to functions reachable from the adapter entry point.
    """
    scopes: list[ast.AST] = [tree]
    allowed = None if reachable_functions is None else set(reachable_functions)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if allowed is not None and node.name not in allowed:
                continue
            scopes.append(node)
    return scopes


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """Yield the nodes belonging to one scope, stopping at nested definitions.

    ``ast.walk`` on a module reaches into every function, which is what put one
    function's local names in another's binding table.
    """
    queue: list[ast.AST] = [scope]
    while queue:
        node = queue.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if child is not scope and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            queue.append(child)


def collect_path_uses(
    module_name: str,
    source_path: Path,
    *,
    reachable_functions: Iterable[str] | None = None,
    reachable_nodes: Iterable[int] | None = None,
) -> list[PathUse]:
    """Return every rooted path expression in one module, classified.

    Two passes per function. The first binds local names to path expressions,
    so ``control_path = artifact_root / CONTROL_PATH`` followed by
    ``control_path.is_file()`` reads as one read of that path. The second
    classifies every occurrence by the call it reaches. A reachable-function
    filter keeps unrelated adapter branches out of the result.
    """
    tree = _parse(source_path)
    if tree is None:
        return []
    constants = _module_constants(tree)
    guards = _GuardMap(tree)
    uses: dict[tuple[int, str, str], PathUse] = {}

    def record(
        root: str,
        relative: str,
        resolved: bool,
        line: int,
        node: ast.AST,
        mode: str,
        *,
        extra_guard: bool = False,
    ) -> None:
        key = (line, root, relative)
        existing = uses.get(key)
        if existing is not None and existing.mode != "unclassified":
            return
        uses[key] = PathUse(
            module=module_name,
            file=source_path,
            line=line,
            root=root,
            relative=relative,
            guarded=extra_guard or guards.is_guarded(node),
            resolved=resolved,
            mode=mode,
        )

    selected_nodes = None if reachable_nodes is None else set(reachable_nodes)
    for scope in _scopes(tree, reachable_functions):
        # name -> (root, relative, resolved, line, guarded). The guard travels
        # with the binding, because `path = artifact_root / ...` inside an
        # `if` followed by `path.is_file()` outside it is one conditional read,
        # and reading the guard off the use site alone calls it unconditional.
        bindings: dict[str, tuple[str, str, bool, int, bool]] = {}
        scope_nodes = list(_scope_nodes(scope))
        if selected_nodes is not None and scope is not tree:
            scope_nodes = [node for node in scope_nodes if id(node) in selected_nodes]
        for node in scope_nodes:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                split = _split_path(node.value, constants)
                if isinstance(target, ast.Name) and split is not None:
                    root, relative, resolved = split
                    bindings[target.id] = (
                        root,
                        relative,
                        resolved,
                        node.lineno,
                        guards.is_guarded(node.value),
                    )

        for node in scope_nodes:
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in READER_CALLS:
                mode = "read"
            elif name in WRITER_CALLS:
                mode = "write"
            else:
                mode = "unclassified"
            operands: list[ast.expr] = list(node.args)
            if isinstance(node.func, ast.Attribute):
                operands.append(node.func.value)
            for operand in operands:
                split = _split_path(operand, constants)
                if split is not None:
                    root, relative, resolved = split
                    record(root, relative, resolved, operand.lineno, operand, mode)
                    continue
                bare = _strip_calls(operand)
                if isinstance(bare, ast.Name) and bare.id in bindings:
                    root, relative, resolved, line, bound_guard = bindings[bare.id]
                    record(root, relative, resolved, line, operand, mode, extra_guard=bound_guard)

        # Anything bound but never classified is still worth reporting, because
        # a path this module builds under the shared root is a path it expects
        # to exist.
        for root, relative, resolved, line, bound_guard in bindings.values():
            key = (line, root, relative)
            if key not in uses:
                uses[key] = PathUse(
                    module=module_name,
                    file=source_path,
                    line=line,
                    root=root,
                    relative=relative,
                    guarded=bound_guard,
                    resolved=resolved,
                    mode="unclassified",
                )
    return list(uses.values())


# --- config keys, folded out of the source -----------------------------------


@dataclass(frozen=True)
class ConfigRead:
    """One config field an adapter requires, with where it was read from."""

    module: str
    file: Path
    line: int
    dotted: str
    guarded: bool
    required: bool
    source: str  # "subscript" or "refusal-message"

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


CONFIG_ROOTS = frozenset(
    {
        "adapters",
        "binder",
        "cofold",
        "controls",
        "filters",
        "generation",
        "optimization",
        "runtime",
        "scoring",
        "selection",
        "sequence_design",
        "stages",
        "targets",
    }
)
CONFIG_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_])((?:"
    + "|".join(sorted(CONFIG_ROOTS))
    + r")(?:\.[A-Za-z0-9_{}\[\]-]+)+)"
)


def _is_integer_index(node: ast.AST) -> bool:
    """True when a subscript key can only index a list.

    `config["targets"][0]["target_id"]` reads the same field a
    `for target in config["targets"]` loop reads, but an integer key used to
    render as the mapping marker `{}`. That marker carries a segment of its own
    while the element marker `[]` does not, so the two forms of the same read
    never lined up and the indexed one vouched for nothing.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return True
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    )


def _config_key_text(node: ast.AST) -> str | None:
    """Fold a static config-key expression into its string value.

    Adapter code sometimes splits a key to keep related field names together.
    The AST still carries both literal segments, so the contract scan resolves
    ``config[\"residue\" + \"_map_sha256\"]`` as one concrete key.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _config_key_text(node.left)
        right = _config_key_text(node.right)
        if left is not None and right is not None:
            return left + right
    if isinstance(node, ast.JoinedStr):
        values = [_config_key_text(value) for value in node.values]
        if all(value is not None for value in values):
            return "".join(value for value in values if value is not None)
    return None


# Package helpers that take a parent mapping and a literal key, return the
# child, and record an error when it is absent or the wrong type.
REQUIRING_ACCESSORS = frozenset({"require_object", "require_list"})

# Package helpers that take one config value and return a view of it.
PASSTHROUGH_ACCESSORS = frozenset({"enabled_items"})

# What such a helper reads off each element to decide the view. `enabled_items`
# filters on `enabled`, but it does so against its own parameter, so nothing
# recorded that read for the caller and every declared `enabled` looked unread.
PASSTHROUGH_ELEMENT_READS: Mapping[str, tuple[str, ...]] = {"enabled_items": ("enabled",)}


def _source_position(node: ast.AST) -> tuple[int, int]:
    """Where this node starts, for nodes that carry a position and those that do not."""
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _copied_value(node: ast.AST) -> ast.AST | None:
    """Return the value a whole-value copy expression copies, if it is one."""
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return None
    function = node.func
    if isinstance(function, ast.Attribute) and function.attr == "deepcopy":
        return node.args[0]
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "loads"
        and isinstance(inner := node.args[0], ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "dumps"
        and len(inner.args) >= 1
    ):
        return inner.args[0]
    return None


def _config_chain(node: ast.AST, roots: Mapping[str, tuple[str, ...]]) -> tuple[tuple[str, ...], bool] | None:
    """Return ``(key path, required)`` for a config access expression."""
    if isinstance(node, ast.Subscript):
        parent = _config_chain(node.value, roots)
        if parent is None:
            return None
        key = node.slice
        key_text = _config_key_text(key)
        if key_text is not None:
            return (*parent[0], key_text), True
        if _is_integer_index(key):
            return (*parent[0], ELEMENT_OF), True
        # The key is a variable, so the audit cannot say which child this is.
        # Returning the parent unchanged pretended the next lookup happened one
        # level up, which turned `controls[group]["id"]` into `controls.id` and
        # reported a field nothing reads. A marker keeps the depth honest.
        marker = "{" + key.id + "}" if isinstance(key, ast.Name) else "{}"
        return (*parent[0], marker), parent[1]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PASSTHROUGH_ACCESSORS
        and len(node.args) == 1
    ):
        # `enabled_items(cofold.get("predictors"))` filters a config list and
        # hands back the same list of records. The chain has to survive the call
        # or every field the caller then reads off an element looks unread.
        return _config_chain(node.args[0], roots)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in REQUIRING_ACCESSORS
        and len(node.args) >= 2
    ):
        # `require_object(config, "scoring", errors)` is this package's own way
        # of reading a config field and refusing without it. It reads as a plain
        # call, so the scan used to stop there, and every field beneath the
        # returned object looked unread. Treat it as the required read it is.
        parent = _config_chain(node.args[0], roots)
        if parent is None:
            return None
        key = node.args[1]
        key_text = _config_key_text(key)
        if key_text is not None:
            return (*parent[0], key_text), True
        marker = "{" + key.id + "}" if isinstance(key, ast.Name) else "{}"
        return (*parent[0], marker), parent[1]
    copied = _copied_value(node)
    if copied is not None:
        # `resolved = json.loads(json.dumps(config))` is how `materialize` takes
        # its working copy. A copy of a config value is that value, so the chain
        # has to survive it or every field read off the copy looks unread.
        return _config_chain(copied, roots)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        parent = _config_chain(node.func.value, roots)
        if parent is None or not node.args:
            return None
        key = node.args[0]
        key_text = _config_key_text(key)
        if key_text is not None:
            return (*parent[0], key_text), False
        marker = "{" + key.id + "}" if isinstance(key, ast.Name) else "{}"
        return (*parent[0], marker), parent[1]
    if isinstance(node, ast.Name):
        prefix = roots.get(node.id)
        return (prefix, True) if prefix is not None else None
    return None


UNKNOWN_KEY = "{}"

_WILDCARD_NAME = re.compile(r"^\{([A-Za-z_][A-Za-z0-9_]*)\}$")

# A variable key expanded from a named collection stays a read of every name in
# it. The cap keeps a large registry from turning one loop into hundreds of
# reads, and a collection over the cap keeps the marker it already had.
KEY_SET_EXPANSION_LIMIT = 32

_EMPTY_KEY_SETS: dict[str, tuple[str, ...]] = {}

CONSTRUCTED_COLLECTIONS = frozenset(
    {"sorted", "tuple", "list", "set", "frozenset", "reversed"}
)


def _literal_string_collection(
    node: ast.AST, collections_table: Mapping[str, tuple[str, ...]]
) -> tuple[str, ...] | None:
    """Return the literal key names a collection expression holds.

    `for field in ("tool_revision", "reference_revision")` reads two named
    config fields through one variable subscript. A variable marker vouches for
    no leaf, which is right for a key the scan cannot name and wrong here, where
    the source names both.
    """
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_config_key_text(item) for item in node.elts]
        if values and all(value is not None for value in values):
            return tuple(value for value in values if value is not None)
        return None
    if isinstance(node, ast.Name):
        return collections_table.get(node.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in CONSTRUCTED_COLLECTIONS
        and len(node.args) == 1
    ):
        return _literal_string_collection(node.args[0], collections_table)
    return None


def _module_key_collections(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Module-level names bound to a literal collection of config key names."""
    table: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        values = _literal_string_collection(node.value, table)
        if values is not None:
            table[target.id] = values
    return table


def _module_key_constants(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Module-level names bound to a single literal config key name.

    `IMAGE_KEY = "CLAUDE_BINDER_MODAL_IMAGE"` then `environment.get(IMAGE_KEY)`
    names one concrete key through one constant. The chain builder sees a Name
    and records the `{IMAGE_KEY}` marker, which vouches for no leaf, so the
    declared key reads as an orphan. Returning the literal as a one-value set
    lets the existing marker expansion resolve it, which names only what the
    source already states.
    """
    table: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        text = _config_key_text(node.value)
        if text is not None:
            table[target.id] = (text,)
    return table


def _expanded_keys(
    keys: Sequence[str], key_sets: Mapping[str, tuple[str, ...]]
) -> list[tuple[str, ...]]:
    """Name the concrete keys a chain reads through a named collection.

    `for name in sorted(threshold_names): thresholds.get(name)` reads each
    threshold the set names. The wildcard form is kept alongside, so this only
    adds what the source states and takes nothing away.
    """
    options: list[tuple[str, ...]] = []
    total = 1
    expanded = False
    for segment in keys:
        match = _WILDCARD_NAME.match(segment)
        values = key_sets.get(match.group(1)) if match is not None else None
        if not values:
            options.append((segment,))
            continue
        options.append(values)
        total *= len(values)
        expanded = True
    if not expanded or total > KEY_SET_EXPANSION_LIMIT:
        return []
    return [tuple(combination) for combination in itertools.product(*options)]


def _merge_chains(
    chains: Sequence[tuple[tuple[str, ...], bool] | None],
) -> tuple[tuple[str, ...], bool] | None:
    """Merge the chains one tuple position takes across a literal sequence.

    `for group_name, group in (("positive", positive), ("negative", negative))`
    binds `group` to two different config lists. They share a parent, so the
    merge keeps the segments they agree on and marks the one they disagree on as
    a variable key. That is the marker a subscript through a variable already
    produces, and it lands on every sibling at that depth.
    """
    if not chains or any(chain is None or not chain[0] for chain in chains):
        return None
    keys = [chain[0] for chain in chains if chain is not None]
    if len({len(key) for key in keys}) != 1:
        return None
    if len({key[0] for key in keys}) != 1:
        return None
    merged = tuple(
        segments[0] if len(set(segments)) == 1 else UNKNOWN_KEY
        for segments in zip(*keys)
    )
    return merged, all(chain[1] for chain in chains if chain is not None)


def _unpacked_chains(
    value: ast.AST,
    roots: Mapping[str, tuple[str, ...]],
    *,
    arity: int,
    iterated: bool,
) -> list[tuple[tuple[str, ...], bool] | None] | None:
    """Return one chain per position of a tuple target, or None when unknown.

    `for index, control in enumerate(group)` is how this package walks a config
    list whenever the body needs the position for a refusal message. The target
    is a tuple, so the single-name binding pass skipped it, and every field the
    body then read off `control` looked unread. That one gap accounted for the
    whole `controls.*` group of orphan declarations.
    """
    if not iterated:
        if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == arity:
            return [_config_chain(item, roots) for item in value.elts]
        return None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id == "enumerate" and value.args and arity == 2:
            inner = _config_chain(value.args[0], roots)
            if inner is None or not inner[0]:
                return None
            return [None, ((*inner[0], ELEMENT_OF), inner[1])]
        if value.func.id == "zip" and len(value.args) == arity:
            positions: list[tuple[tuple[str, ...], bool] | None] = []
            for argument in value.args:
                inner = _config_chain(argument, roots)
                positions.append(
                    None
                    if inner is None or not inner[0]
                    else ((*inner[0], ELEMENT_OF), inner[1])
                )
            return positions
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "items"
        and not value.args
        and arity == 2
    ):
        # `for name, spec in config["adapters"].items()` names no key, so the
        # value side carries the same variable-key marker a `[name]` subscript
        # produces. The key side names no config field at all.
        inner = _config_chain(value.func.value, roots)
        if inner is None or not inner[0]:
            return None
        return [None, ((*inner[0], UNKNOWN_KEY), inner[1])]
    if isinstance(value, (ast.Tuple, ast.List)):
        rows: list[list[tuple[tuple[str, ...], bool] | None]] = []
        for element in value.elts:
            if not isinstance(element, (ast.Tuple, ast.List)) or len(element.elts) != arity:
                return None
            rows.append([_config_chain(item, roots) for item in element.elts])
        if not rows:
            return None
        return [_merge_chains([row[index] for row in rows]) for index in range(arity)]
    return None


def _access_root_name(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        return _access_root_name(node.func.value)
    return node.id if isinstance(node, ast.Name) else None



_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _comprehension_roots(
    scope_nodes: Sequence[ast.AST],
    roots_at: Callable[[tuple[int, int]], Mapping[str, tuple[str, ...]]],
) -> dict[int, dict[str, tuple[str, ...]]]:
    """Map each node inside a comprehension to the roots that apply there.

    A comprehension target is scoped to its comprehension, the way a function
    local is scoped to its function. Binding it at function level made `item` in
    one comprehension answer for `item` in the next, which invented required
    fields nothing reads. The comprehension still sees the names around it, so
    its table starts from the enclosing one.
    """
    scoped: dict[int, dict[str, tuple[str, ...]]] = {}
    comprehensions = [node for node in scope_nodes if isinstance(node, _COMPREHENSIONS)]
    # Outermost first, so a nested comprehension inherits the one holding it.
    comprehensions.sort(key=_source_position)
    for comp in comprehensions:
        local = dict(scoped.get(id(comp)) or roots_at(_source_position(comp)))
        for generator in comp.generators:
            chain = _config_chain(generator.iter, local)
            if isinstance(generator.target, ast.Name) and chain is not None and chain[0]:
                local[generator.target.id] = (*chain[0], ELEMENT_OF)
        for node in ast.walk(comp):
            scoped.setdefault(id(node), local)
    return scoped


CONFIG_CALL_PROPAGATION_DEPTH = 1


def _scope_config_roots(
    scope: ast.AST,
    base_roots: Mapping[str, tuple[str, ...]],
    selected_nodes: set[int] | None,
) -> tuple[
    list[ast.AST],
    set[str],
    Callable[[tuple[int, int]], dict[str, tuple[str, ...]]],
]:
    """Resolve local config aliases and return the roots visible at each position."""
    optional_roots: set[str] = set()
    scope_nodes = list(_scope_nodes(scope))
    if selected_nodes is not None and not isinstance(scope, ast.Module):
        scope_nodes = [node for node in scope_nodes if id(node) in selected_nodes]

    bindings: list[tuple[ast.expr, ast.expr, bool]] = []
    for node in scope_nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            bindings.append((node.targets[0], node.value, False))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            # A loop over a config list binds its element to that list, so
            # `for target in config["targets"]` makes `target["site"]` read
            # as `targets[].site`. The marker records that the read happens
            # once per element, which is also to say never when there are
            # none.
            bindings.append((node.target, node.iter, True))
    # `_scope_nodes` does not yield in source order, so a name bound from
    # another bound name could be visited before its parent was known and
    # then never resolve. Order the pass by source position and repeat it
    # until the table stops growing, so an alias chain resolves whatever
    # order the walk happens to produce.
    #
    # A binding answers only for the reads that follow it. A function is
    # free to reuse a name, as `validate_campaign` does with `contract`,
    # which is a filter contract in one loop and a parameter-range contract
    # in another. One chain for the whole scope let the first use answer for
    # the second, which invented fields under the wrong parent and hid the
    # real ones.
    bindings = [item for item in bindings if isinstance(item[0], (ast.Name, ast.Tuple))]
    bindings.sort(key=lambda item: _source_position(item[0]))
    binding_positions = [_source_position(item[0]) for item in bindings]
    resolved: dict[int, list[tuple[str, tuple[str, ...]]]] = {}

    def _tables() -> list[dict[str, tuple[str, ...]]]:
        """Return one roots table per binding, holding the preceding roots."""
        table = dict(base_roots)
        out: list[dict[str, tuple[str, ...]]] = []
        for index in range(len(bindings)):
            out.append(table)
            names = resolved.get(index)
            if names:
                table = dict(table)
                for name, keys in names:
                    table[name] = keys
        out.append(table)
        return out

    while True:
        tables = _tables()
        known = len(resolved)
        for index, (target, value, iterated) in enumerate(bindings):
            if index in resolved:
                continue
            view = tables[index]
            if isinstance(target, ast.Tuple):
                positions = _unpacked_chains(
                    value, view, arity=len(target.elts), iterated=iterated
                )
                # A position resolves only once the name it reads through is
                # bound, which can take a later round. Recording an empty
                # result here would retire the binding before its turn.
                if positions is None or not any(
                    item is not None and item[0] for item in positions
                ):
                    continue
                names = [
                    (element.id, position[0])
                    for element, position in zip(target.elts, positions)
                    if isinstance(element, ast.Name)
                    and position is not None
                    and position[0]
                ]
                resolved[index] = names
                continue
            chain = _config_chain(value, view)
            # An empty chain is the config root itself, which is what
            # `resolved = json.loads(json.dumps(config))` binds. Treating it
            # as unresolved left every field read off the working copy
            # unread. Only `None` means the scan could not follow the value.
            if chain is None:
                continue
            keys = (*chain[0], ELEMENT_OF) if iterated else chain[0]
            resolved[index] = [(target.id, keys)]
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "get"
                and len(value.args) >= 2
            ):
                optional_roots.add(target.id)
        if len(resolved) == known:
            break
    tables = _tables()

    def _roots_at(position: tuple[int, int]) -> dict[str, tuple[str, ...]]:
        return tables[bisect.bisect_left(binding_positions, position)]

    return scope_nodes, optional_roots, _roots_at


def _call_actuals(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    call: ast.Call,
) -> dict[str, ast.AST]:
    """Match statically positioned call arguments to one local function signature."""
    positional = [*function.args.posonlyargs, *function.args.args]
    actuals: dict[str, ast.AST] = {}
    for parameter, argument in zip(positional, call.args):
        if isinstance(argument, ast.Starred):
            break
        actuals[parameter.arg] = argument
    parameters = {parameter.arg for parameter in [*positional, *function.args.kwonlyargs]}
    for keyword in call.keywords:
        if keyword.arg in parameters:
            actuals[keyword.arg] = keyword.value
    return actuals


def _one_hop_call_contexts(
    tree: ast.Module,
    scopes: Sequence[ast.AST],
    selected_nodes: set[int] | None,
) -> dict[int, list[tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]]]:
    """Bind direct local callees from config chains resolved in their callers.

    The prepass uses only roots resolved inside each caller. It never consumes
    the contexts it produces, which enforces CONFIG_CALL_PROPAGATION_DEPTH and
    leaves a chain passed through two helper calls unresolved.
    """
    if CONFIG_CALL_PROPAGATION_DEPTH != 1:
        raise AssertionError("the config call propagation implementation supports exactly one hop")
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    selected_scopes = {id(scope) for scope in scopes}
    contexts: dict[
        int,
        list[tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]],
    ] = {}
    seen: dict[
        int,
        set[
            tuple[
                tuple[tuple[str, tuple[str, ...]], ...],
                tuple[tuple[str, tuple[str, ...]], ...],
            ]
        ],
    ] = {}
    for caller in scopes:
        scope_nodes, _, roots_at = _scope_config_roots(
            caller,
            {"config": (), "campaign": ()},
            selected_nodes,
        )
        for node in scope_nodes:
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
                and id(functions[node.func.id]) in selected_scopes
            ):
                continue
            root_bindings: dict[str, tuple[str, ...]] = {}
            literal_bindings: dict[str, tuple[str, ...]] = {}
            for parameter, argument in _call_actuals(functions[node.func.id], node).items():
                chain = _config_chain(argument, roots_at(_source_position(node)))
                if chain is not None:
                    root_bindings[parameter] = chain[0]
                literal = _config_key_text(argument)
                if literal is not None:
                    literal_bindings[parameter] = (literal,)
            if not root_bindings:
                continue
            context_key = (
                tuple(sorted(root_bindings.items())),
                tuple(sorted(literal_bindings.items())),
            )
            target_id = id(functions[node.func.id])
            if context_key in seen.setdefault(target_id, set()):
                continue
            seen[target_id].add(context_key)
            contexts.setdefault(target_id, []).append((root_bindings, literal_bindings))
    return contexts


def _collect_scope_config_reads(
    *,
    module_name: str,
    source_path: Path,
    tree: ast.Module,
    scope: ast.AST,
    selected_nodes: set[int] | None,
    module_key_collections: Mapping[str, tuple[str, ...]],
    module_key_constants: Mapping[str, tuple[str, ...]],
    raising_names: set[str],
    guards: _GuardMap,
    reads: dict[tuple[str, int], ConfigRead],
    propagated_roots: Mapping[str, tuple[str, ...]],
    propagated_literals: Mapping[str, tuple[str, ...]],
) -> None:
    """Collect one scope under one direct caller context."""
    base_roots: dict[str, tuple[str, ...]] = {
        "config": (),
        "campaign": (),
        **propagated_roots,
    }
    scope_nodes, optional_roots, roots_at = _scope_config_roots(
        scope,
        base_roots,
        selected_nodes,
    )

    # Names this scope iterates over a collection of literal key names.
    assigned_names: set[str] = set()
    for node in scope_nodes:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            assigned_names.add(node.targets[0].id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
    active_literals = {
        name: values
        for name, values in propagated_literals.items()
        if name not in assigned_names
    }
    collections_table = dict(module_key_collections)
    collections_table.update(active_literals)
    for node in scope_nodes:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            values = _literal_string_collection(node.value, collections_table)
            if values is not None:
                collections_table.setdefault(node.targets[0].id, values)
    # A key set belongs to the loop that iterates it, not to the function.
    # `validate_campaign` reuses `name` for the liability rules, the threshold
    # set and the registered checks, so one table per scope would conflate the
    # three sets.
    key_set_scopes: dict[int, dict[str, tuple[str, ...]]] = {}
    loops: list[tuple[tuple[int, int], str, ast.expr, list[ast.AST]]] = []
    for node in scope_nodes:
        if isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.target, ast.Name):
            loops.append((_source_position(node), node.target.id, node.iter, list(node.body)))
        elif isinstance(node, _COMPREHENSIONS):
            loops.extend(
                (_source_position(node), generator.target.id, generator.iter, [node])
                for generator in node.generators
                if isinstance(generator.target, ast.Name)
            )
    # Outermost first, so a nested loop overrides the one holding it.
    loops.sort(key=lambda item: item[0])
    for _, name, iterable, body in loops:
        values = _literal_string_collection(iterable, collections_table)
        if values is None:
            continue
        for statement in body:
            for inner in ast.walk(statement):
                key_set_scopes.setdefault(id(inner), {})[name] = values

    scoped_roots = _comprehension_roots(scope_nodes, roots_at)
    for node in scope_nodes:
        node_roots = scoped_roots.get(id(node), roots_at(_source_position(node)))
        chain = (
            _config_chain(node, node_roots)
            if isinstance(node, (ast.Subscript, ast.Call))
            else None
        )
        if chain is None or len(chain[0]) < 1:
            continue
        keys, required = chain
        has_default = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and len(node.args) >= 2
        )
        if _access_root_name(node) in optional_roots:
            required = False
        elif not required and not has_default:
            required = _bound_name(node, tree) in raising_names
        # A name this scope rebinds is no longer the module constant, so the
        # constant table answers only for names the scope leaves alone.
        node_key_sets = {
            name: values
            for name, values in module_key_constants.items()
            if name not in assigned_names
        }
        node_key_sets.update(active_literals)
        node_key_sets.update(key_set_scopes.get(id(node), _EMPTY_KEY_SETS))
        element_fields = (
            PASSTHROUGH_ELEMENT_READS.get(node.func.id, ())
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            else ()
        )
        for field in element_fields:
            element_dotted = _dotted((*keys, ELEMENT_OF, field))
            element_key = (element_dotted, node.lineno)
            if element_key not in reads:
                reads[element_key] = ConfigRead(
                    module=module_name,
                    file=source_path,
                    line=node.lineno,
                    dotted=element_dotted,
                    guarded=guards.is_guarded(node),
                    required=False,
                    source="subscript",
                )
        for variant in (keys, *_expanded_keys(keys, node_key_sets)):
            dotted = _dotted(variant)
            key = (dotted, node.lineno)
            if key in reads:
                continue
            reads[key] = ConfigRead(
                module=module_name,
                file=source_path,
                line=node.lineno,
                dotted=dotted,
                guarded=guards.is_guarded(node),
                required=required,
                source="subscript",
            )


def collect_config_reads(
    module_name: str,
    source_path: Path,
    *,
    reachable_functions: Iterable[str] | None = None,
    reachable_nodes: Iterable[int] | None = None,
) -> list[ConfigRead]:
    """Return every config field the module names, and whether it refuses without it.

    A ``config["a"]["b"]`` subscript is required by construction. A
    ``.get("b")`` is required only when the module raises on the result, which
    is the pattern every adapter here uses for a field it cannot proceed
    without. A reachable-function filter keeps unrelated adapter branches out
    of the result.
    """
    tree = _parse(source_path)
    if tree is None:
        return []
    guards = _GuardMap(tree)
    reads: dict[tuple[str, int], ConfigRead] = {}
    raising_names = _names_that_raise(tree, reachable_functions, reachable_nodes)

    scopes = _scopes(tree, reachable_functions)
    selected_nodes = None if reachable_nodes is None else set(reachable_nodes)
    module_key_collections = _module_key_collections(tree)
    module_key_constants = _module_key_constants(tree)
    for scope in scopes:
        _collect_scope_config_reads(
            module_name=module_name,
            source_path=source_path,
            tree=tree,
            scope=scope,
            selected_nodes=selected_nodes,
            module_key_collections=module_key_collections,
            module_key_constants=module_key_constants,
            raising_names=raising_names,
            guards=guards,
            reads=reads,
            propagated_roots={},
            propagated_literals={},
        )

    call_contexts = _one_hop_call_contexts(tree, scopes, selected_nodes)
    for scope in scopes:
        for propagated_roots, propagated_literals in call_contexts.get(id(scope), []):
            _collect_scope_config_reads(
                module_name=module_name,
                source_path=source_path,
                tree=tree,
                scope=scope,
                selected_nodes=selected_nodes,
                module_key_collections=module_key_collections,
            module_key_constants=module_key_constants,
                raising_names=raising_names,
                guards=guards,
                reads=reads,
                propagated_roots=propagated_roots,
                propagated_literals=propagated_literals,
            )

    for scope in scopes:
        scope_nodes = list(_scope_nodes(scope))
        if selected_nodes is not None and scope is not tree:
            scope_nodes = [node for node in scope_nodes if id(node) in selected_nodes]
        for node in scope_nodes:
            text = _literal_text(node)
            if text is None:
                continue
            marker = "config is missing required field: "
            if marker not in text:
                continue
            tail = text.split(marker, 1)[1].strip().strip(".")
            dotted = tail.split(" ", 1)[0].strip().strip(".")
            if not dotted or "{" in dotted.split(".")[0]:
                continue
            key = (dotted, node.lineno)
            if key in reads:
                continue
            reads[key] = ConfigRead(
                module=module_name,
                file=source_path,
                line=node.lineno,
                dotted=dotted,
                guarded=False,
                required=True,
                source="refusal-message",
            )
    return list(reads.values())


def collect_literal_config_reads(
    module_name: str,
    source_path: Path,
    *,
    reachable_functions: Iterable[str] | None = None,
    reachable_nodes: Iterable[int] | None = None,
) -> list[ConfigRead]:
    """Return config keys named in literal error and refusal messages.

    A dynamic lookup such as ``runtime.get(key)`` can hide the concrete key.
    Adapters preserve that key in a refusal message or a roster declaration.
    Reading those literals closes that gap without treating every optional
    ``dict.get`` call as a required profile field.
    """
    tree = _parse(source_path)
    if tree is None:
        return []
    selected_nodes = None if reachable_nodes is None else set(reachable_nodes)
    reads: dict[tuple[str, int], ConfigRead] = {}
    for scope in _scopes(tree, reachable_functions):
        for node in _scope_nodes(scope):
            if selected_nodes is not None and scope is not tree and id(node) not in selected_nodes:
                continue
            text = _literal_text(node)
            if text is None:
                continue
            lowered = text.lower()
            if not (
                "config is missing" in lowered
                or "config value is missing" in lowered
                or ("runtime." in lowered and "missing" in lowered)
            ):
                continue
            for match in CONFIG_LITERAL_RE.finditer(text):
                dotted = match.group(1)
                reads.setdefault(
                    (dotted, node.lineno),
                    ConfigRead(
                        module=module_name,
                        file=source_path,
                        line=node.lineno,
                        dotted=dotted,
                        guarded=False,
                        required=True,
                        source="literal",
                    ),
                )
    return list(reads.values())


def _literal_text(node: ast.AST) -> str | None:
    """Return the text of a string literal, keeping an f-string's placeholders.

    A refusal that names a field usually interpolates the last part, as in
    ``filters.metric_sources.{filter_id}``. Reading only the constant prefix
    truncates that to ``filters.metric_sources``, which the config does carry,
    so the refusal the adapter would actually raise went unreported.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
            parts.append("{" + value.value.id + "}")
        else:
            parts.append("{}")
    return "".join(parts)


def _condition_is_false_when_none(node: ast.AST, name: str) -> bool:
    """Return whether ``node`` cannot enter its branch when ``name`` is absent.

    This is deliberately conservative. It recognizes presence guards without
    trying to evaluate arbitrary Python. A field checked only when present is
    optional, even if that check raises for an unsupported supplied value.
    """
    if isinstance(node, ast.Name) and node.id == name:
        return True
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return any(_condition_is_false_when_none(value, name) for value in node.values)
        if isinstance(node.op, ast.Or):
            return all(_condition_is_false_when_none(value, name) for value in node.values)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        left, right = node.left, node.comparators[0]
        left_is_name = isinstance(left, ast.Name) and left.id == name
        right_is_name = isinstance(right, ast.Name) and right.id == name
        left_is_none = isinstance(left, ast.Constant) and left.value is None
        right_is_none = isinstance(right, ast.Constant) and right.value is None
        if isinstance(node.ops[0], (ast.IsNot, ast.NotEq)):
            return (left_is_name and right_is_none) or (right_is_name and left_is_none)
        if isinstance(node.ops[0], (ast.Is, ast.Eq)):
            return (left_is_name and not right_is_none) or (right_is_name and not left_is_none)
    return False


def _names_that_raise(
    tree: ast.Module,
    reachable_functions: Iterable[str] | None = None,
    reachable_nodes: Iterable[int] | None = None,
) -> set[str]:
    """Return names tested in an ``if`` whose body raises."""
    names: set[str] = set()
    selected_nodes = None if reachable_nodes is None else set(reachable_nodes)
    for scope in _scopes(tree, reachable_functions):
        scope_nodes = list(_scope_nodes(scope))
        if selected_nodes is not None and scope is not tree:
            scope_nodes = [node for node in scope_nodes if id(node) in selected_nodes]
        for node in scope_nodes:
            if not isinstance(node, ast.If):
                continue
            if not any(
                isinstance(inner, ast.Raise)
                for inner in ast.walk(ast.Module(body=node.body, type_ignores=[]))
            ):
                continue
            for inner in ast.walk(node.test):
                if (
                    isinstance(inner, ast.Name)
                    and not _condition_is_false_when_none(node.test, inner.id)
                ):
                    names.add(inner.id)
    return names


_BOUND_NAMES: dict[int, dict[int, str]] = {}


def _bound_names(tree: ast.Module) -> dict[int, str]:
    """Map each assigned expression to the name it is bound to, once per tree."""
    cached = _BOUND_NAMES.get(id(tree))
    if cached is not None:
        return cached
    mapping: dict[int, str] = {}
    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                mapping[id(statement.value)] = target.id
    _BOUND_NAMES[id(tree)] = mapping
    return mapping


def _bound_name(node: ast.AST, tree: ast.Module) -> str:
    """Return the variable one expression is assigned to, when there is one."""
    return _bound_names(tree).get(id(node), "")


# --- emitted row keys --------------------------------------------------------


@dataclass(frozen=True)
class EmissionFacts:
    """What one adapter's source says about the row fields it produces."""

    keys: frozenset[str]
    mentioned: frozenset[str]
    passthrough: tuple[str, ...]
    offpath_imports: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """False when part of the row builder lives where the audit cannot read it."""
        return not self.offpath_imports


def collect_emission_facts(source_path: Path) -> EmissionFacts:
    """Return the field names a module writes, names it mentions, and its copies.

    ``keys`` are strings written as a dict key: a dict literal, a
    ``row["field"] = value`` assignment or a ``setdefault``. A list of field
    names is not counted, because a list is how this codebase declares a
    contract rather than how it emits one.

    ``mentioned`` is every string constant in the module. A field the adapter
    never names anywhere is a field it has no notion of, and that is the only
    absence this audit calls certain enough to gate on.

    ``passthrough`` records the places the module copies a mapping wholesale,
    which is ``{**row, ...}``, ``dict(row)`` or ``row.copy()``. A module that
    does this can carry a field it never names, so its absences are reported as
    smells.
    """
    tree = _parse(source_path)
    if tree is None:
        return EmissionFacts(frozenset(), frozenset(), ())
    keys: set[str] = set()
    mentioned: set[str] = set()
    passthrough: list[str] = []
    offpath: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "path":
            if isinstance(node.value, ast.Name) and node.value.id == "sys":
                offpath.append(f"{source_path}:{node.lineno} extends sys.path before importing")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            mentioned.add(node.value)
        if isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
                elif key is None:
                    passthrough.append(f"{source_path}:{node.lineno} spreads a mapping")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    if isinstance(target.slice.value, str):
                        keys.add(target.slice.value)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name == "setdefault" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    keys.add(first.value)
            elif name == "copy" and not node.args:
                passthrough.append(f"{source_path}:{node.lineno} copies a mapping")
            elif name == "dict" and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                passthrough.append(f"{source_path}:{node.lineno} copies a mapping")
    return EmissionFacts(
        frozenset(keys), frozenset(mentioned), tuple(passthrough), tuple(sorted(set(offpath)))
    )


def collect_emitted_keys(source_path: Path) -> set[str]:
    """Return every string the module writes as a dict key."""
    return set(collect_emission_facts(source_path).keys)


# --- module resolution -------------------------------------------------------


def reset_caches() -> None:
    """Drop the parsed-module and write-index caches.

    The caches are keyed by path and an audit is normally a one-shot process,
    so this exists for a test that rewrites a file at the same path between
    cases.
    """
    _TREES.clear()
    _WRITE_INDEX.clear()
    _BOUND_NAMES.clear()
    _MODULE_CLOSURE_CACHE.clear()
    _STAGE_CONFIG_READ_CACHE.clear()


def _package_root(explicit: Path | None = None) -> Path:
    return (explicit or Path(__file__).resolve().parent).resolve()


def _module_path(module_name: str, package_root: Path) -> Path | None:
    """Return the source file for a dotted ``claude_binder`` module name."""
    if not module_name.startswith("claude_binder"):
        return None
    parts = module_name.split(".")[1:]
    if not parts:
        candidate = package_root / "__main__.py"
        return candidate if candidate.is_file() else None
    candidate = package_root.joinpath(*parts).with_suffix(".py")
    if candidate.is_file():
        return candidate
    package_init = package_root.joinpath(*parts) / "__init__.py"
    return package_init if package_init.is_file() else None


def _entry_module(argv: Sequence[str]) -> str | None:
    """Return the module an argv template runs under ``python -m``."""
    for index, token in enumerate(argv):
        if token == "-m" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _module_import_name(module_name: str) -> str:
    """Return the import target that Python uses for an argv module name."""
    return f"{module_name}.__main__" if module_name == "claude_binder" else module_name


def _import_adapter_module(module_name: str, package_root: Path) -> ModuleType:
    """Import an adapter module, loading a synthetic-audit source as a fallback."""
    import_name = _module_import_name(module_name)
    try:
        return importlib.import_module(import_name)
    except ImportError as import_error:
        source = _module_path(module_name, package_root)
        if source is None:
            raise import_error
        synthetic_name = f"_contract_audit_{abs(hash(source.resolve())):x}"
        specification = importlib.util.spec_from_file_location(synthetic_name, source)
        if specification is None or specification.loader is None:
            raise import_error
        module = importlib.util.module_from_spec(specification)
        sys.modules[synthetic_name] = module
        try:
            specification.loader.exec_module(module)
        except Exception:
            sys.modules.pop(synthetic_name, None)
            raise
        return module


def _builder_argument(module: ModuleType, name: str) -> object | None:
    """Return a safe parser-builder argument that an adapter exports."""
    if name != "arm":
        return None
    for candidate in ("ARM_FAST", "ARM_FULL"):
        value = getattr(module, candidate, None)
        if value is not None:
            return value
    return None


def _call_builder(
    builder: Callable[..., Any], module: ModuleType
) -> argparse.ArgumentParser | None:
    """Call a parser builder when its required values are exported constants."""
    try:
        parameters = inspect.signature(builder).parameters.values()
    except (TypeError, ValueError):
        return None
    arguments: list[object] = []
    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ) or parameter.default is not inspect.Parameter.empty:
            continue
        value = _builder_argument(module, parameter.name)
        if value is None:
            return None
        arguments.append(value)
    parser = builder(*arguments)
    return parser if isinstance(parser, argparse.ArgumentParser) else None


def _captured_parser(factory: Callable[[], Any]) -> argparse.ArgumentParser | None:
    """Capture an argparse parser built by a legacy parse or main function."""
    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser

    class CapturingArgumentParser(original):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            captured.append(self)

    factory_globals = getattr(factory, "__globals__", None)
    if not isinstance(factory_globals, dict) or factory_globals.get("argparse") is not argparse:
        return None

    class ArgparseProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(argparse, name)

    proxy = ArgparseProxy()
    proxy.ArgumentParser = CapturingArgumentParser
    original_factory_argparse = factory_globals["argparse"]
    original_argv = sys.argv
    try:
        # Python 3.14 resolves ArgumentParser inside ArgumentParser.__init__ at
        # call time. Rebinding argparse.ArgumentParser would make its explicit
        # super(ArgumentParser, self) call re-enter the same initializer. The
        # factory sees the capturing proxy, while argparse itself keeps its
        # original class for the cooperative superclass call.
        factory_globals["argparse"] = proxy
        sys.argv = ["contract-audit"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                factory()
            except SystemExit:
                pass
            except Exception:
                # Legacy adapters often build the parser inline and then call a
                # local parse helper. A missing helper means this source exposes
                # no parser contract for static inspection.
                pass
    finally:
        sys.argv = original_argv
        factory_globals["argparse"] = original_factory_argparse
    return captured[0] if captured else None


def _argument_parser(module: ModuleType) -> argparse.ArgumentParser | None:
    """Return the parser an adapter exposes without dispatching its command."""
    builder = getattr(module, "build_parser", None)
    if callable(builder):
        parser = _call_builder(builder, module)
        if parser is not None:
            return parser

    base = getattr(module, "esmfold2_predictor", None)
    base_builder = getattr(base, "build_parser", None)
    if isinstance(base, ModuleType) and callable(base_builder):
        parser = _call_builder(base_builder, module)
        if parser is not None:
            return parser

    for name in ("parse_arguments", "main", "cli"):
        factory = getattr(module, name, None)
        if callable(factory):
            parser = _captured_parser(factory)
            if parser is not None:
                return parser
    return None


def _selected_parsers(
    parser: argparse.ArgumentParser, command_tokens: Sequence[str]
) -> list[argparse.ArgumentParser]:
    """Return the root parser and each subparser selected by the command argv."""
    selected = [parser]
    active = parser
    token_index = 0
    while True:
        subparsers = next(
            (
                action
                for action in active._actions
                if isinstance(action, argparse._SubParsersAction)
            ),
            None,
        )
        if subparsers is None:
            return selected
        match_index = next(
            (
                index
                for index in range(token_index, len(command_tokens))
                if command_tokens[index] in subparsers.choices
            ),
            None,
        )
        if match_index is None:
            return selected
        active = subparsers.choices[command_tokens[match_index]]
        selected.append(active)
        token_index = match_index + 1


def _required_parser_flags(
    parser: argparse.ArgumentParser, command_tokens: Sequence[str]
) -> set[str]:
    """Return every required option on the parser branch selected by argv."""
    required: set[str] = set()
    for selected in _selected_parsers(parser, command_tokens):
        for action in selected._actions:
            if action.required and action.option_strings:
                required.add(next((flag for flag in action.option_strings if flag.startswith("--")), action.option_strings[0]))
    return required


def _template_validation_flags(module: ModuleType, command: str) -> set[str]:
    """Return flags an adapter declares its own pre-dispatch validation requires."""
    provider = getattr(module, "template_required_arguments", None)
    if not callable(provider):
        return set()
    values = provider(command)
    if not isinstance(values, (list, tuple, set, frozenset)) or any(
        not isinstance(value, str) or not value.startswith("--") for value in values
    ):
        raise ValueError("template_required_arguments must return command flags")
    return set(values)


def _template_flags(argv: Sequence[str]) -> set[str]:
    """Return option flags present in an argv template."""
    return {
        token.partition("=")[0]
        for token in argv
        if isinstance(token, str) and token.startswith("-") and token != "-m"
    }


def check_adapter_argument_templates(
    plan: Mapping[str, Any], report: AuditReport, *, package_root: Path
) -> None:
    """Require every bound adapter template to satisfy its parser and own validation."""
    for adapter_id, adapter in sorted(_bound_adapters(plan).items()):
        argv = adapter.get("command_argv_template")
        if not isinstance(argv, list) or any(not isinstance(token, str) for token in argv):
            report.skipped.append(f"{adapter_id}: command_argv_template is not a string argv list")
            continue
        module_name = _entry_module(argv)
        if module_name is None:
            report.skipped.append(f"{adapter_id}: command_argv_template names no Python module")
            continue
        module_index = argv.index("-m") + 2
        command = argv[module_index] if module_index < len(argv) else ""
        stages = _stages_for_adapter(plan, adapter_id)
        stage = stages[0] if stages else "<unbound>"
        location = f"run-plan.json adapters[{adapter_id}].command_argv_template"
        try:
            module = _import_adapter_module(module_name, package_root)
            parser = _argument_parser(module)
            if parser is None:
                report.skipped.append(f"{adapter_id}: {module_name} exposes no inspectable argparse parser")
                continue
            required = _required_parser_flags(parser, argv[module_index:])
            validation_required = _template_validation_flags(module, command)
        except ModuleNotFoundError as exc:
            if _module_path(module_name, package_root) is not None:
                report.skipped.append(
                    f"{adapter_id}: {module_name} parser import needs unavailable module {exc.name}"
                )
                continue
            report.add(
                Finding(
                    stage=stage,
                    adapter=adapter_id,
                    location=location,
                    field="command_argv_template",
                    problem=f"could not import {module_name}: {exc}",
                    fix="point command_argv_template at an importable adapter module",
                    severity=RUN_FAILING,
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001
            report.add(
                Finding(
                    stage=stage,
                    adapter=adapter_id,
                    location=location,
                    field="command_argv_template",
                    problem=f"could not inspect {module_name}: {type(exc).__name__}: {exc}",
                    fix="make the adapter parser importable and expose its validation-required flags",
                    severity=RUN_FAILING,
                )
            )
            continue
        report.checked.append(f"{adapter_id}: argv parser {module_name}")
        present = _template_flags(argv)
        for flag in sorted(required | validation_required):
            if flag in present:
                continue
            source = "parser requires" if flag in required else "adapter validation requires"
            report.add(
                Finding(
                    stage=stage,
                    adapter=adapter_id,
                    location=location,
                    field=f"command_argv_template[{flag}]",
                    problem=f"{source} {flag}, but the template omits it",
                    fix=f"add {flag} and its declared value to command_argv_template",
                    severity=RUN_FAILING,
                )
            )


def _first_party_imports(source_path: Path, module_name: str) -> set[str]:
    """Return every ``claude_binder`` module this one imports, at any depth in the file."""
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError):
        return set()
    package = module_name.rsplit(".", 1)[0] if "." in module_name else module_name
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("claude_binder"):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package
                for _ in range(node.level - 1):
                    base = base.rsplit(".", 1)[0] if "." in base else base
                prefix = f"{base}.{node.module}" if node.module else base
            elif node.module and node.module.startswith("claude_binder"):
                prefix = node.module
            else:
                continue
            found.add(prefix)
            for alias in node.names:
                found.add(f"{prefix}.{alias.name}")
    return found


def module_closure(entry_module: str, package_root: Path, *, depth: int | None = 1) -> dict[str, Path]:
    """Return the entry module and the first-party modules it imports.

    Shared library modules are dropped unless the entry names one. See
    ``LIBRARY_MODULES`` for why.
    """
    cache_key = (entry_module, package_root.resolve(), depth)
    cached = _MODULE_CLOSURE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    resolved: dict[str, Path] = {}
    frontier = [entry_module]
    levels = None if depth is None else depth + 1
    level = 0
    while frontier and (levels is None or level < levels):
        next_frontier: list[str] = []
        for name in frontier:
            if name in resolved:
                continue
            if name != entry_module and name in LIBRARY_MODULES:
                continue
            path = _module_path(name, package_root)
            if path is None:
                continue
            resolved[name] = path
            next_frontier.extend(_first_party_imports(path, name))
        frontier = next_frontier
        level += 1
    _MODULE_CLOSURE_CACHE[cache_key] = dict(resolved)
    return resolved


# --- adapter entry-point reachability ---------------------------------------


_UNKNOWN_SELECTOR = object()


def _argv_value(argv: Sequence[str], flag: str) -> str | None:
    for index, token in enumerate(argv[:-1]):
        if token == flag:
            value = argv[index + 1]
            return value if isinstance(value, str) and "{{" not in value else None
    return None


def _argv_command(argv: Sequence[str]) -> str | None:
    try:
        module_index = argv.index("-m")
    except ValueError:
        return None
    for token in argv[module_index + 2 :]:
        if isinstance(token, str) and not token.startswith("--") and "{{" not in token:
            return token
    return None


def _selectors(
    argv: Sequence[str],
    *,
    stage_id: str | None,
    adapter_id: str | None,
) -> dict[str, str]:
    return {
        "command": _argv_command(argv) or "",
        "stage": stage_id or _argv_value(argv, "--stage") or "",
        "adapter_id": adapter_id or _argv_value(argv, "--adapter-id") or "",
    }


def _selector_value(node: ast.AST, values: Mapping[str, str]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
        return values.get(node.attr, _UNKNOWN_SELECTOR)
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        items = [_selector_value(item, values) for item in node.elts]
        if any(item is _UNKNOWN_SELECTOR for item in items):
            return _UNKNOWN_SELECTOR
        if isinstance(node, ast.Tuple):
            return tuple(items)
        if isinstance(node, ast.Set):
            return set(items)
        return items
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        value = _selector_value(node.operand, values)
        return _UNKNOWN_SELECTOR if value is _UNKNOWN_SELECTOR else not bool(value)
    if isinstance(node, ast.BoolOp):
        items = [_selector_value(item, values) for item in node.values]
        if isinstance(node.op, ast.And):
            if any(item is False for item in items):
                return False
            return True if all(item is True for item in items) else _UNKNOWN_SELECTOR
        if any(item is True for item in items):
            return True
        return False if all(item is False for item in items) else _UNKNOWN_SELECTOR
    if isinstance(node, ast.IfExp):
        test = _selector_value(node.test, values)
        if test is True:
            return _selector_value(node.body, values)
        if test is False:
            return _selector_value(node.orelse, values)
        return _UNKNOWN_SELECTOR
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _selector_value(node.func.value, values)
        args = [_selector_value(item, values) for item in node.args]
        if base is _UNKNOWN_SELECTOR or any(item is _UNKNOWN_SELECTOR for item in args):
            return _UNKNOWN_SELECTOR
        if node.func.attr in {"startswith", "endswith"} and len(args) == 1:
            method = getattr(base, node.func.attr, None)
            return method(args[0]) if callable(method) else _UNKNOWN_SELECTOR
        if node.func.attr in {"removeprefix", "removesuffix"} and len(args) == 1:
            method = getattr(base, node.func.attr, None)
            return method(args[0]) if callable(method) else _UNKNOWN_SELECTOR
        return _UNKNOWN_SELECTOR
    if isinstance(node, ast.Compare):
        left = _selector_value(node.left, values)
        result: list[bool] = []
        for operation, comparator_node in zip(node.ops, node.comparators):
            right = _selector_value(comparator_node, values)
            if left is _UNKNOWN_SELECTOR or right is _UNKNOWN_SELECTOR:
                return _UNKNOWN_SELECTOR
            if isinstance(operation, ast.Eq):
                result.append(left == right)
            elif isinstance(operation, ast.NotEq):
                result.append(left != right)
            elif isinstance(operation, ast.In):
                result.append(left in right)
            elif isinstance(operation, ast.NotIn):
                result.append(left not in right)
            elif isinstance(operation, ast.Is):
                result.append(left is right)
            elif isinstance(operation, ast.IsNot):
                result.append(left is not right)
            else:
                return _UNKNOWN_SELECTOR
            left = right
        return all(result)
    return _UNKNOWN_SELECTOR


class _SelectedCallVisitor(ast.NodeVisitor):
    """Collect calls from branches selected by the command argv."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = values
        self.calls: list[ast.Call] = []
        self.nodes: set[int] = set()

    def generic_visit(self, node: ast.AST) -> None:
        self.nodes.add(id(node))
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_block(self, body: Sequence[ast.stmt]) -> None:
        for statement in body:
            if isinstance(statement, ast.If):
                exits = self._visit_if(statement)
            else:
                self.visit(statement)
                exits = isinstance(statement, (ast.Return, ast.Raise, ast.Continue, ast.Break))
            if exits:
                return

    def _visit_block_with_exit(self, body: Sequence[ast.stmt]) -> bool:
        for statement in body:
            if isinstance(statement, ast.If):
                exits = self._visit_if(statement)
            else:
                self.visit(statement)
                exits = isinstance(statement, (ast.Return, ast.Raise, ast.Continue, ast.Break))
            if exits:
                return True
        return False

    def _visit_if(self, node: ast.If) -> bool:
        self.nodes.add(id(node))
        self.visit(node.test)
        outcome = _selector_value(node.test, self.values)
        if outcome is True:
            return self._visit_block_with_exit(node.body)
        elif outcome is False:
            return self._visit_block_with_exit(node.orelse)
        body_exits = self._visit_block_with_exit(node.body)
        else_exits = self._visit_block_with_exit(node.orelse)
        return body_exits and else_exits

    def visit_If(self, node: ast.If) -> None:
        self._visit_if(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.nodes.add(id(node))
        self.visit(node.test)
        outcome = _selector_value(node.test, self.values)
        if outcome is True:
            self.visit(node.body)
        elif outcome is False:
            self.visit(node.orelse)
        else:
            self.visit(node.body)
            self.visit(node.orelse)


def _top_level_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _import_targets(source_path: Path, module_name: str) -> dict[str, tuple[str, str | None]]:
    tree = _parse(source_path)
    if tree is None:
        return {}
    package = module_name.rsplit(".", 1)[0] if "." in module_name else module_name
    targets: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("claude_binder"):
                    continue
                local = alias.asname or alias.name.rsplit(".", 1)[-1]
                targets[local] = (alias.name, None)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = package
            for _ in range(node.level - 1):
                base = base.rsplit(".", 1)[0] if "." in base else base
            prefix = f"{base}.{node.module}" if node.module else base
        elif node.module and node.module.startswith("claude_binder"):
            prefix = node.module
        else:
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            local = alias.asname or alias.name
            if node.module is None:
                targets[local] = (f"{prefix}.{alias.name}", None)
            else:
                targets[local] = (prefix, alias.name)
    return targets


def _entry_function_names(
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
    command: str | None,
) -> list[str]:
    if "main" in functions:
        return ["main"]
    if "cli" in functions:
        return ["cli"]
    preferred = {
        "run": ("run", "run_stage", "execute", "execute_stage"),
        "parse": ("parse", "parse_stage", "parse_execution_stage"),
        "toolcheck": ("toolcheck",),
        "rank": ("rank",),
    }.get(command or "", ())
    for name in preferred:
        if name in functions:
            return [name]
    if len(functions) == 1:
        return [next(iter(functions))]
    return sorted(functions)


def _analysis_entry(entry: str) -> str:
    """Resolve package entry points whose dispatch lives in the CLI module."""

    if entry == "claude_binder":
        return "claude_binder.lane"
    return entry


@dataclass(frozen=True)
class _Reachability:
    functions: Mapping[str, frozenset[str]]
    nodes: Mapping[str, frozenset[int]]


def reachable_functions(
    entry_module: str,
    closure: Mapping[str, Path],
    argv: Sequence[str],
    *,
    stage_id: str | None = None,
    adapter_id: str | None = None,
) -> _Reachability:
    """Return functions reachable for one adapter command and stage.

    The walk follows local calls and first-party imported calls. Branches whose
    conditions resolve from ``argv`` are pruned, while unknown conditions keep
    both sides for recall.
    """
    trees = {
        module_name: _parse(source)
        for module_name, source in closure.items()
    }
    functions = {
        module_name: _top_level_functions(tree)
        for module_name, tree in trees.items()
        if tree is not None
    }
    imports = {
        module_name: _import_targets(source, module_name)
        for module_name, source in closure.items()
    }
    selected = _selectors(argv, stage_id=stage_id, adapter_id=adapter_id)
    command = selected["command"] or None
    result: dict[str, set[str]] = {module_name: set() for module_name in closure}
    selected_nodes: dict[str, set[int]] = {module_name: set() for module_name in closure}
    queue: list[tuple[str, str]] = []
    for name in _entry_function_names(functions.get(entry_module, {}), command):
        queue.append((entry_module, name))

    while queue:
        module_name, function_name = queue.pop()
        if function_name in result.setdefault(module_name, set()):
            continue
        node = functions.get(module_name, {}).get(function_name)
        if node is None:
            continue
        result[module_name].add(function_name)
        visitor = _SelectedCallVisitor(selected)
        visitor.visit_block(node.body)
        selected_nodes.setdefault(module_name, set()).update(visitor.nodes)
        for call in visitor.calls:
            target_module: str | None = None
            target_function: str | None = None
            if isinstance(call.func, ast.Name):
                local_name = call.func.id
                if local_name in functions.get(module_name, {}):
                    target_module, target_function = module_name, local_name
                elif local_name in imports.get(module_name, {}):
                    target_module, target_function = imports[module_name][local_name]
            elif isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
                imported = imports.get(module_name, {}).get(call.func.value.id)
                if imported is not None:
                    target_module, target_function = imported[0], call.func.attr
            if target_module in functions and target_function in functions[target_module]:
                queue.append((target_module, target_function))
    return _Reachability(
        functions={module_name: frozenset(names) for module_name, names in result.items()},
        nodes={module_name: frozenset(nodes) for module_name, nodes in selected_nodes.items()},
    )


# --- the plan's own declarations ---------------------------------------------


@dataclass(frozen=True)
class DeclaredOutput:
    stage_id: str
    ordinal: int
    artifact_id: str
    publish_path: str | None
    path_template: str | None
    kind: str
    minimum_count: int | None
    records_per_count: int | None
    required_fields: tuple[str, ...]


def _ordered_stages(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    stages = [item for item in plan.get("stages", []) if isinstance(item, Mapping)]
    order = plan.get("ordered_stage_ids")
    if not isinstance(order, list):
        return stages
    position = {str(value): index for index, value in enumerate(order)}
    return sorted(stages, key=lambda item: position.get(str(item.get("stage_id")), len(position)))


def declared_outputs(plan: Mapping[str, Any]) -> list[DeclaredOutput]:
    outputs: list[DeclaredOutput] = []
    for ordinal, stage in enumerate(_ordered_stages(plan)):
        for raw in stage.get("outputs", []) or []:
            if not isinstance(raw, Mapping):
                continue
            outputs.append(
                DeclaredOutput(
                    stage_id=str(stage.get("stage_id")),
                    ordinal=ordinal,
                    artifact_id=str(raw.get("artifact_id")),
                    publish_path=raw.get("publish_path") if isinstance(raw.get("publish_path"), str) else None,
                    path_template=raw.get("path_template") if isinstance(raw.get("path_template"), str) else None,
                    kind=str(raw.get("kind", "")),
                    minimum_count=raw.get("minimum_count") if isinstance(raw.get("minimum_count"), int) else None,
                    records_per_count=(
                        raw.get("records_per_count") if isinstance(raw.get("records_per_count"), int) else None
                    ),
                    required_fields=tuple(
                        str(value) for value in (raw.get("required_fields") or []) if isinstance(value, str)
                    ),
                )
            )
    return outputs


def _publish_index(outputs: Iterable[DeclaredOutput]) -> dict[str, DeclaredOutput]:
    index: dict[str, DeclaredOutput] = {}
    for output in outputs:
        if output.publish_path:
            index.setdefault(_normalize(output.publish_path), output)
    return index


def _dotted(keys: Sequence[str]) -> str:
    """Render a key path, attaching the element marker to the key it iterates."""
    text = ""
    for key in keys:
        if key == ELEMENT_OF:
            text += ELEMENT_OF
        elif text:
            text += f".{key}"
        else:
            text = key
    return text


def _normalize(value: str) -> str:
    return str(PurePosixPath(value.strip().strip("/")))


def _find_producer(
    published: Mapping[str, DeclaredOutput], relative: str
) -> DeclaredOutput | None:
    """Return the stage output that publishes this path.

    A path the audit could not fully fold carries a wildcard where the value
    goes, so `optimization/rounds/round-*/decision.json` has to match the
    concrete `round-1` path the plan publishes. Comparing those as strings
    reported a producer that is right there in the plan as missing.
    """
    exact = published.get(relative)
    if exact is not None or "*" not in relative:
        return exact
    matches = [path for path in published if fnmatch(path, relative)]
    if not matches:
        return None
    return published[min(matches, key=lambda path: published[path].ordinal)]


def _bound_adapters(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the adapters a stage actually runs, keyed by adapter id."""
    registered = {
        str(item.get("adapter_id")): item
        for item in plan.get("adapters", [])
        if isinstance(item, Mapping)
    }
    bound: dict[str, Mapping[str, Any]] = {}
    for stage in _ordered_stages(plan):
        adapter_id = str(stage.get("adapter_id"))
        if adapter_id in registered:
            bound[adapter_id] = registered[adapter_id]
    return bound


def _stages_for_adapter(plan: Mapping[str, Any], adapter_id: str) -> list[str]:
    return [
        str(stage.get("stage_id"))
        for stage in _ordered_stages(plan)
        if str(stage.get("adapter_id")) == adapter_id
    ]


def _render(token: str, context: Mapping[str, Any]) -> str:
    """Substitute the plan's own context into an argv template token."""
    rendered = token
    for key, value in context.items():
        rendered = rendered.replace("{{" + str(key) + "}}", str(value))
    while "{{" in rendered and "}}" in rendered:
        head, rest = rendered.split("{{", 1)
        _, tail = rest.split("}}", 1)
        rendered = f"{head}*{tail}"
    return rendered


# --- check one: undeclared reads ---------------------------------------------


_WRITE_INDEX: dict[Path, dict[str, str]] = {}


def _write_index(package_root: Path) -> dict[str, str]:
    """Return every file basename the package writes, mapped to where.

    A read with no declared producer is more actionable when the finding can
    name the code that would have produced it, so this is a hint for the fix
    line rather than a check of its own. The scan reads every module once and
    the result is held for the process, because an audit asks for it repeatedly.
    """
    cached = _WRITE_INDEX.get(package_root)
    if cached is not None:
        return cached
    index: dict[str, str] = {}
    for source in sorted(package_root.rglob("*.py")):
        if "tests" in source.parts:
            continue
        for use in collect_path_uses(source.stem, source):
            if use.mode != "write":
                continue
            name = PurePosixPath(use.relative).name
            if name and "*" not in name:
                index.setdefault(name, f"{source}:{use.line}")
    _WRITE_INDEX[package_root] = index
    return index


def _basename_producer(package_root: Path, basename: str) -> str | None:
    return _write_index(package_root).get(basename)


def check_undeclared_reads(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: AuditReport,
    *,
    package_root: Path,
) -> None:
    """Report every read of the shared artifact root with no declared producer."""
    outputs = declared_outputs(plan)
    published = _publish_index(outputs)
    context = plan.get("context") if isinstance(plan.get("context"), Mapping) else {}
    bound = _bound_adapters(plan)
    seen_config_fields: set[tuple[str, str]] = set()

    for adapter_id, adapter in sorted(bound.items()):
        stages = _stages_for_adapter(plan, adapter_id)
        first_stage = stages[0] if stages else "<unbound>"
        earliest = min(
            (output.ordinal for output in outputs if output.stage_id in stages),
            default=len(outputs),
        )

        # Reads named directly in the plan's own argv. Two of the nine lived
        # only here and appear in no adapter source. These stay per-adapter
        # whatever the entry module is, because the plan names them against one
        # adapter rather than inside shared code.
        for template_key in ("command_argv_template", "parser_argv_template", "toolcheck_argv"):
            argv = adapter.get(template_key)
            if not isinstance(argv, list):
                continue
            for index, token in enumerate(argv):
                if not isinstance(token, str) or "{{artifact_root}}/" not in token:
                    continue
                flag = argv[index - 1] if index > 0 and isinstance(argv[index - 1], str) else ""
                if flag in WRITE_FLAGS:
                    continue
                relative = _normalize(_render(token.split("{{artifact_root}}/", 1)[1], context))
                producer = _find_producer(published, relative)
                if producer is not None and (producer.ordinal < earliest or producer.stage_id in stages):
                    continue
                report.add(
                    _undeclared_finding(
                        stage=first_stage,
                        adapter=adapter_id,
                        location=f"run-plan.json adapters[{adapter_id}].{template_key}[{index}]",
                        subject=f"{flag} {relative}".strip(),
                        relative=relative,
                        producer=producer,
                        earliest=earliest,
                        guarded=False,
                        resolved="*" not in relative,
                        package_root=package_root,
                        published=published,
                    )
                )

    for adapter_id, adapter in sorted(bound.items()):
        stages = _stages_for_adapter(plan, adapter_id)
        first_stage = stages[0] if stages else "<unbound>"
        earliest = min(
            (output.ordinal for output in outputs if output.stage_id in stages),
            default=len(outputs),
        )
        entry = _entry_module(adapter.get("command_argv_template") or [])
        if entry is None:
            report.skipped.append(f"{adapter_id}: command_argv_template names no module")
            continue
        analysis_entry = _analysis_entry(entry)
        closure = module_closure(analysis_entry, package_root, depth=None)
        if not closure:
            report.skipped.append(f"{adapter_id}: could not resolve {entry} under {package_root}")
            continue
        report.checked.append(f"{adapter_id}: {', '.join(sorted(closure))}")
        dropped = sorted(
            name
            for name in _first_party_imports(closure[analysis_entry], analysis_entry) & LIBRARY_MODULES
            if name not in closure
        )
        for name in dropped:
            report.skipped.append(
                f"{adapter_id}: {name} is a shared library, so its paths are not read as this adapter's contract"
            )

        for stage_id in stages:
            reachable = reachable_functions(
                analysis_entry,
                closure,
                adapter.get("command_argv_template") or [],
                stage_id=stage_id,
                adapter_id=adapter_id,
            )
            for module_name, source in sorted(closure.items()):
                selected_functions = reachable.functions.get(module_name, frozenset())
                selected_nodes = reachable.nodes.get(module_name, frozenset())
                for use in collect_path_uses(
                    module_name,
                    source,
                    reachable_functions=selected_functions,
                    reachable_nodes=selected_nodes,
                ):
                    if use.root != SHARED_ROOT or use.mode == "write":
                        continue
                    if not use.relative:
                        continue
                    if _round_path_is_unreachable(config, stage_id, use):
                        continue
                    producer = _find_producer(published, _normalize(use.relative))
                    if producer is not None and (producer.ordinal < earliest or producer.stage_id in stages):
                        continue
                    if producer is not None and producer.ordinal >= earliest:
                        report.add(
                            Finding(
                                stage=first_stage,
                                adapter=adapter_id,
                                location=use.location,
                                field=f"artifact_root/{use.relative}",
                                problem=(
                                    f"the only stage that declares this path is {producer.stage_id}, "
                                    f"which the plan orders at position {producer.ordinal}, "
                                    f"after {first_stage} at position {earliest}"
                                ),
                                fix=(
                                    f"move {producer.stage_id} before {first_stage} in ordered_stage_ids, "
                                    f"or add {first_stage} to its depends_on"
                                ),
                                severity=SMELL if use.guarded or not use.resolved else RUN_FAILING,
                                confidence=CERTAIN if use.resolved else UNCERTAIN,
                            )
                        )
                        continue
                    report.add(
                        _undeclared_finding(
                            stage=first_stage,
                            adapter=adapter_id,
                            location=use.location,
                            subject=f"artifact_root/{use.relative}",
                            relative=use.relative,
                            producer=None,
                            earliest=earliest,
                            guarded=use.guarded,
                            resolved=use.resolved,
                            package_root=package_root,
                            published=published,
                        )
                    )

                for read in sorted(
                    collect_config_reads(
                        module_name,
                        source,
                        reachable_functions=selected_functions,
                        reachable_nodes=selected_nodes,
                    ),
                    key=lambda item: item.line,
                ):
                    if not read.required:
                        continue
                    if not _config_read_is_reachable(config, read.dotted):
                        continue
                    if _resolved_config_supplies_read(config, read.dotted):
                        continue
                    # One missing field is one finding. An adapter names the same
                    # field on every line it refuses over, and the refusals differ
                    # only past the part the audit cannot resolve, so the concrete
                    # prefix is what identifies the problem.
                    concrete = _concrete_prefix(read.dotted)
                    if (adapter_id, concrete) in seen_config_fields:
                        continue
                    seen_config_fields.add((adapter_id, concrete))
                    shortfall = (
                        f"{concrete} is empty, so no key inside it resolves"
                        if concrete != read.dotted
                        else "the resolved config does not carry it"
                    )
                    report.add(
                        Finding(
                            stage=first_stage,
                            adapter=adapter_id,
                            location=read.location,
                            field=read.dotted,
                            problem=f"{adapter_id} requires config field {read.dotted} and {shortfall}",
                            fix=(
                                f"set {read.dotted} in the profile that composes this config, "
                                f"or make {module_name} treat it as optional"
                            ),
                            severity=SMELL if read.guarded else RUN_FAILING,
                            confidence=CERTAIN if read.source == "subscript" else UNCERTAIN,
                        )
                    )

    _check_bundled_inputs(config, plan, report)
    _check_relative_config_paths(config, plan, report, package_root=package_root)


EXECUTOR_PRIVATE_ARTIFACT_PREFIXES = frozenset(
    {
        "artifact-index.json",
        "executed-commands.jsonl",
        "history",
        "receipts",
        "spend.jsonl",
        "stage-progress.jsonl",
        "stages",
        "validation",
    }
)


def _is_executor_private_artifact_path(relative: str) -> bool:
    """Return whether a path belongs to executor bookkeeping rather than a stage output."""
    first_part = PurePosixPath(relative).parts[0] if relative else ""
    return first_part in EXECUTOR_PRIVATE_ARTIFACT_PREFIXES


def check_executor_artifact_reads(
    plan: Mapping[str, Any],
    report: AuditReport,
    *,
    package_root: Path,
) -> None:
    """Require each lane runtime artifact read to name a declared stage output.

    The executor validates stage output after an adapter returns. Those validators
    consume the shared artifact directory too, so adapter-only analysis leaves a
    pre-spend gap. The collector also treats a validator call that receives a
    path as a reader. That preserves the path through helpers such as
    ``validate_filter_cohort``.
    """
    source = package_root / "lane.py"
    if not source.is_file():
        report.skipped.append("lane runtime reads: lane.py is unavailable under the package root")
        return
    published = _publish_index(declared_outputs(plan))
    report.checked.append("lane runtime artifact reads")
    for use in collect_path_uses("claude_binder.lane", source):
        if use.root != SHARED_ROOT or use.mode != "read" or not use.relative:
            continue
        relative = _normalize(use.relative)
        if _is_executor_private_artifact_path(relative):
            continue
        producer = _find_producer(published, relative)
        if producer is not None:
            continue
        report.add(
            _undeclared_finding(
                stage="lane-runtime",
                adapter="claude-binder-lane",
                location=use.location,
                subject=f"artifact_root/{relative}",
                relative=relative,
                producer=None,
                earliest=len(declared_outputs(plan)),
                guarded=use.guarded,
                resolved=use.resolved,
                package_root=package_root,
                published=published,
            )
        )


def _enabled_filter_contracts(config: Mapping[str, Any], stage_id: str) -> list[Mapping[str, Any]]:
    filters = config.get("filters")
    if not isinstance(filters, Mapping):
        return []
    disabled_raw = filters.get("disabled_checks")
    disabled = set(disabled_raw) if isinstance(disabled_raw, list) else set()
    contracts = filters.get("contracts")
    if not isinstance(contracts, list):
        return []
    return [
        contract
        for contract in contracts
        if isinstance(contract, Mapping)
        and contract.get("stage_id") == stage_id
        and contract.get("filter_id") not in disabled
    ]


def _filter_lineage_finding(
    *,
    consumer: DeclaredOutput,
    source_stage: str,
    artifact_id: str,
    producer: DeclaredOutput | None,
) -> Finding:
    reference = f"{source_stage}:{artifact_id}"
    if producer is None:
        problem = f"filter-lineage validation reads {reference}, and no stage declares that output"
        fix = f"declare {reference} with a publish_path, or remove the runtime consumer"
    else:
        problem = (
            f"filter-lineage validation reads {reference} from position {producer.ordinal}, "
            f"after its consumer {consumer.stage_id} at position {consumer.ordinal}"
        )
        fix = f"move {source_stage} before {consumer.stage_id} in ordered_stage_ids"
    return Finding(
        stage=consumer.stage_id,
        adapter="claude-binder-lane",
        location="claude_binder/lane.py:declared_filter_cohort_paths",
        field=reference,
        problem=problem,
        fix=fix,
        severity=RUN_FAILING,
    )


def check_filter_lineage_contract_closure(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: AuditReport,
) -> None:
    """Validate the declared outputs read by every filter-lineage consumer."""
    outputs = declared_outputs(plan)
    by_reference = {(output.stage_id, output.artifact_id): output for output in outputs}
    consumers = {output.stage_id: output for output in outputs if output.artifact_id == "screen-score-table"}
    if "score-screen" not in consumers:
        return
    consumer = consumers["score-screen"]
    required = [
        ("normalize-candidates", "normalized-candidates"),
        ("filter-integrity", "integrity-passing-candidates"),
        ("filter-novelty", "passing-candidates"),
    ]
    if _enabled_filter_contracts(config, "filter-integrity"):
        required.append(("filter-integrity", "integrity-filter-observations"))
    if _enabled_filter_contracts(config, "filter-novelty"):
        required.append(("filter-novelty", "novelty-filter-observations"))
    for source_stage, artifact_id in required:
        producer = by_reference.get((source_stage, artifact_id))
        if producer is None or producer.ordinal >= consumer.ordinal:
            report.add(
                _filter_lineage_finding(
                    consumer=consumer,
                    source_stage=source_stage,
                    artifact_id=artifact_id,
                    producer=producer,
                )
            )


def _relative_path_values(value: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    """Yield every ``(dotted key, value)`` in the config that reads as a relative file."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _relative_path_values(item, f"{prefix}.{key}" if prefix else str(key))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _relative_path_values(item, f"{prefix}[{index}]")
        return
    if not isinstance(value, str) or not prefix.split(".")[-1].split("[")[0].endswith("path"):
        return
    if value.startswith("/") or not PurePosixPath(value).suffix:
        return
    yield prefix, value


def _check_relative_config_paths(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: AuditReport,
    *,
    package_root: Path,
) -> None:
    """Report a config key naming a relative file that resolves nowhere.

    A model roster ledger was read through ``runtime.model_roster_path`` and
    written by nothing. The reader treated the gap as a check it could not
    perform rather than a refusal, so the run went ahead with the roster
    unverified. A relative path that resolves to neither a packaged file nor a
    stage output is that same hole.
    """
    published = {_normalize(path) for path in _publish_index(declared_outputs(plan))}
    declared_artifacts = {
        f"{output.stage_id}:{output.artifact_id}"
        for output in declared_outputs(plan)
    }
    # `paths.schema_file` resolves a packaged name under `data/`, so a config
    # value naming a schema is relative to that rather than to the package root.
    bases = (package_root, package_root / "data")
    seen: set[str] = set()
    for dotted, value in _relative_path_values(config):
        if _metric_source_fallback_is_unreachable(config, dotted, declared_artifacts):
            continue
        if _normalize(value) in published or value in seen:
            continue
        if any((base / value).exists() for base in bases):
            continue
        seen.add(value)
        report.add(
            Finding(
                stage="<config>",
                adapter="<resolved config>",
                location=f"config.resolved.json {dotted}",
                field=dotted,
                problem=(
                    f"{dotted} names {value}, and that is neither a file under the package nor "
                    f"a path any stage publishes"
                ),
                fix=(
                    f"ship {value} as package data, or declare it as a stage output publish_path, "
                    f"or make the reader refuse instead of skipping the check"
                ),
                severity=SMELL,
            )
        )


# --- check three: declared input and shipped config provenance ---------------


SPECIAL_INPUT_TYPES = {
    "run-bundle": "run-bundle",
    "run-bundle:controls": "control-structures",
    "stage-receipts": "stage-receipts",
}


def check_declared_inputs(plan: Mapping[str, Any], report: AuditReport) -> None:
    """Check every stage input against an earlier declared output.

    The executor accepts three external input classes. All other inputs must
    name a stage and artifact that the materialized plan publishes earlier.
    """
    stages = _ordered_stages(plan)
    positions = {str(stage.get("stage_id")): index for index, stage in enumerate(stages)}
    outputs = {
        (output.stage_id, output.artifact_id): output
        for output in declared_outputs(plan)
    }
    stage_ids = set(positions)
    for stage in stages:
        stage_id = str(stage.get("stage_id"))
        stage_position = positions[stage_id]
        for index, input_ref in enumerate(stage.get("inputs", []) or []):
            location = f"run-plan.json stages[{stage_id}].inputs[{index}]"
            if not isinstance(input_ref, str):
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=str(stage.get("adapter_id")),
                        location=location,
                        field=f"stages[{stage_id}].inputs[{index}]",
                        problem="the declared input is not a stage artifact reference",
                        fix="use an external input name or stage-id:artifact-id",
                        severity=RUN_FAILING,
                    )
                )
                continue
            if input_ref in SPECIAL_INPUT_TYPES:
                continue
            if ":" not in input_ref:
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=str(stage.get("adapter_id")),
                        location=location,
                        field=input_ref,
                        problem="the declared input has no producer stage",
                        fix="declare a bundled input or use stage-id:artifact-id for an upstream output",
                        severity=RUN_FAILING,
                    )
                )
                continue
            source_stage, artifact_id = input_ref.split(":", 1)
            producer = outputs.get((source_stage, artifact_id))
            if source_stage not in stage_ids or producer is None:
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=str(stage.get("adapter_id")),
                        location=location,
                        field=input_ref,
                        problem="the declared input names an artifact that no stage produces",
                        fix="add the producer stage and output, or remove this input",
                        severity=RUN_FAILING,
                    )
                )
                continue
            if producer.ordinal >= stage_position:
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=str(stage.get("adapter_id")),
                        location=location,
                        field=input_ref,
                        problem=(
                            f"the declared input producer {source_stage} runs at position "
                            f"{producer.ordinal}, after {stage_id} at position {stage_position}"
                        ),
                        fix=f"move {source_stage} before {stage_id} and add the dependency",
                        severity=RUN_FAILING,
                    )
                )


def _canonical_config_key(value: str) -> str:
    if value.endswith("["):
        value += "]"
    value = re.sub(r"\[[^\]]*\]", "[]", value)
    value = re.sub(r"\{[^}]*\}", "{}", value)
    return value.strip(".")


def _flatten_config_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if prefix:
        keys.add(_canonical_config_key(prefix))
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            keys.update(_flatten_config_keys(item, child))
    elif isinstance(value, list):
        marker = f"{prefix}[]" if prefix else "[]"
        keys.add(_canonical_config_key(marker))
        for item in value:
            keys.update(_flatten_config_keys(item, marker))
    return keys


def _add_config_source(
    source_map: dict[str, set[str]],
    value: Any,
    source: Path,
    prefix: str = "",
) -> None:
    for key in _flatten_config_keys(value, prefix):
        source_map.setdefault(key, set()).add(str(source))


def _profile_config_sources(
    path: Path,
    source_map: dict[str, set[str]],
    visited: set[Path] | None = None,
) -> None:
    visited = set() if visited is None else visited
    path = path.resolve()
    if path in visited or not path.is_file():
        return
    visited.add(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(document, Mapping):
        return
    base = document.get("base_profile")
    if isinstance(base, str) and base:
        _profile_config_sources(path.parent / base, source_map, visited)

    for key, value in document.items():
        if key not in {"schema_version", "template_id", "base_profile", "overlay"}:
            _add_config_source(source_map, value, path, key)
    overlay = document.get("overlay")
    if not isinstance(overlay, Mapping):
        return
    for block_name in ("top_level", "campaign_overrides"):
        block = overlay.get(block_name)
        if isinstance(block, Mapping):
            _add_config_source(source_map, block, path)

    for item in overlay.get("stage_overrides", []) or []:
        if isinstance(item, Mapping):
            _add_config_source(source_map, item, path, "stages[]")
    for item in overlay.get("adapter_overrides", []) or []:
        if isinstance(item, Mapping):
            _add_config_source(source_map, item, path, "adapters[]")


def _shipped_config_sources(template_root: Path) -> dict[str, set[str]]:
    source_map: dict[str, set[str]] = {}
    campaign_paths = sorted(
        path
        for path in template_root.rglob("*.json")
        if path.name.startswith("campaign")
    )
    for path in campaign_paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        _add_config_source(source_map, document, path)
    profile_root = template_root / "profiles"
    for path in sorted(profile_root.glob("*.json")):
        _profile_config_sources(path, source_map)
    return source_map


def _config_key_covered(read: str, source_keys: Mapping[str, set[str]]) -> bool:
    canonical = _canonical_config_key(read)
    if canonical in source_keys:
        return True
    parts = canonical.split(".")
    for index, part in enumerate(parts):
        if part in {"[]", "{}"}:
            parent = ".".join(parts[:index])
            if parent and parent in source_keys:
                return True
    prefix = canonical.split("{}", 1)[0].rstrip(".")
    if prefix and any(key == prefix or key.startswith(prefix + ".") for key in source_keys):
        return True
    return False


def _config_reads_for_stage(
    stage: Mapping[str, Any],
    adapter: Mapping[str, Any],
    package_root: Path,
) -> list[ConfigRead]:
    commands = (
        (adapter.get("command_argv_template") or [], "command"),
        (adapter.get("parser_argv_template") or [], "parser"),
    )
    cache_key = (
        package_root.resolve(),
        str(stage.get("stage_id", "")),
        str(adapter.get("adapter_id", "")),
        json.dumps(commands, sort_keys=True, separators=(",", ":"), default=str),
    )
    cached = _STAGE_CONFIG_READ_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    reads: list[ConfigRead] = []
    for argv, _label in commands:
        if not isinstance(argv, list):
            continue
        entry = _entry_module(argv)
        if entry is None:
            continue
        analysis_entry = _analysis_entry(entry)
        closure = module_closure(analysis_entry, package_root, depth=None)
        if not closure:
            continue
        reachable = reachable_functions(
            analysis_entry,
            closure,
            argv,
            stage_id=str(stage.get("stage_id")),
            adapter_id=str(adapter.get("adapter_id")),
        )
        for module_name, source in sorted(closure.items()):
            selected_functions = reachable.functions.get(module_name, frozenset())
            selected_nodes = reachable.nodes.get(module_name, frozenset())
            reads.extend(
                item
                for item in collect_config_reads(
                    module_name,
                    source,
                    reachable_functions=selected_functions,
                    reachable_nodes=selected_nodes,
                )
                if item.required
            )
            reads.extend(
                collect_literal_config_reads(
                    module_name,
                    source,
                    reachable_functions=selected_functions,
                    reachable_nodes=selected_nodes,
                )
            )
    _STAGE_CONFIG_READ_CACHE[cache_key] = tuple(reads)
    return reads


def check_profile_config_coverage(
    plan: Mapping[str, Any],
    report: AuditReport,
    *,
    config: Mapping[str, Any] | None = None,
    package_root: Path,
    template_root: Path,
) -> None:
    """Require each adapter's required config key in a shipped source file."""
    source_keys = _shipped_config_sources(template_root)
    has_campaign = any(
        path.name.startswith("campaign") for path in template_root.rglob("*.json")
    )
    has_profile = any((template_root / "profiles").glob("*.json"))
    if not has_campaign and not has_profile:
        report.skipped.append(f"config coverage: no campaign or profile JSON under {template_root}")
        return
    bound = _bound_adapters(plan)
    seen: set[tuple[str, str]] = set()
    for stage in _ordered_stages(plan):
        stage_id = str(stage.get("stage_id"))
        adapter_id = str(stage.get("adapter_id"))
        adapter = bound.get(adapter_id)
        if adapter is None:
            continue
        for read in _config_reads_for_stage(stage, adapter, package_root):
            if config is not None and not _config_read_is_reachable(config, read.dotted):
                continue
            if config is not None and _resolved_config_supplies_read(config, read.dotted):
                continue
            canonical = _canonical_config_key(read.dotted)
            if canonical.startswith("{}") or ".{}" in canonical:
                continue
            key = (adapter_id, canonical)
            if key in seen or _config_key_covered(read.dotted, source_keys):
                continue
            seen.add(key)
            report.add(
                Finding(
                    stage=stage_id,
                    adapter=adapter_id,
                    location=read.location,
                    field=read.dotted,
                    problem=(
                        f"{adapter_id} reads config key {read.dotted}, and no shipped "
                        "profile or campaign template sets it"
                    ),
                    fix=f"set {read.dotted} in a shipped profile or campaign template",
                    severity=RUN_FAILING,
                )
            )


def _undeclared_finding(
    *,
    stage: str,
    adapter: str,
    location: str,
    subject: str,
    relative: str,
    producer: DeclaredOutput | None,
    earliest: int,
    guarded: bool,
    resolved: bool,
    package_root: Path,
    published: Mapping[str, DeclaredOutput],
) -> Finding:
    basename = PurePosixPath(relative).name
    hint = _basename_producer(package_root, basename) if "*" not in basename else None
    near = sorted(
        path for path in published if PurePosixPath(path).name == basename
    )
    # The reader usually names the artifact it wants and guesses the directory,
    # so a declared artifact whose id matches this filename's stem is the
    # likeliest intended target and the most useful thing to put in the fix.
    stem = PurePosixPath(basename).stem
    by_id = sorted(
        f"{output.stage_id}:{output.artifact_id} at {path}"
        for path, output in published.items()
        if output.artifact_id == stem
    )
    if near:
        fix = (
            f"the plan publishes this file at {near[0]}, so point the reader there "
            f"or change the producing stage's publish_path to {relative}"
        )
    elif by_id:
        fix = (
            f"the plan declares {by_id[0]}, so point the reader at that published path "
            f"or change the producing stage's publish_path to {relative}"
        )
    elif hint:
        fix = (
            f"{hint} writes a file of this name, so add the stage that runs it to the plan "
            f"and declare {relative} as its publish_path"
        )
    else:
        fix = (
            f"add a stage that declares {relative} as an output publish_path, "
            f"or publish it as a bundled input before the run starts"
        )
    return Finding(
        stage=stage,
        adapter=adapter,
        location=location,
        field=subject,
        problem=(
            f"{adapter} reads this path and no stage ordered before position {earliest} "
            f"declares it as an output"
        ),
        fix=fix,
        severity=SMELL if guarded or not resolved else RUN_FAILING,
        confidence=CERTAIN if resolved else UNCERTAIN,
    )


def _concrete_prefix(dotted: str) -> str:
    """Return the part of a dotted key before the first value the audit cannot read."""
    parts: list[str] = []
    for part in dotted.split("."):
        if "{" in part or "[" in part:
            break
        parts.append(part)
    return ".".join(parts) or dotted


def _config_has(config: Any, dotted: str) -> bool:
    """True when the resolved config carries this dotted field.

    A key containing a template marker names a value the audit cannot know, so
    the walk stops there and answers yes. A list is descended element by
    element, because ``for target in config["targets"]`` reads a field off
    every target and the dotted path cannot say which one.
    """
    parts = [part for part in dotted.split(".") if part]
    if not parts:
        return True
    head, rest = parts[0], ".".join(parts[1:])
    bare = head.split("[")[0].split("{")[0]
    unresolved = "{" in head or "[" in head
    if isinstance(config, list):
        return not config or all(_config_has(item, dotted) for item in config)
    if head.endswith(ELEMENT_OF):
        container = config.get(bare) if isinstance(config, Mapping) else None
        if not isinstance(container, list) or not container:
            # No elements, so the loop body never runs and nothing inside it is
            # required. An absent container reads the same way, because that is
            # what an adapter iterating `config.get(name, [])` sees.
            return True
        return all(_config_has(item, rest) for item in container)
    if not bare:
        # The whole part is a template, so the key is not knowable. An empty
        # container cannot supply it whatever the key turns out to be, and
        # `filters.metric_sources` sitting empty while a filter demands a source
        # from it is exactly that.
        return bool(config) if isinstance(config, (Mapping, list)) else True
    if not isinstance(config, Mapping) or bare not in config:
        return False
    value = config[bare]
    if unresolved:
        return bool(value) if isinstance(value, (Mapping, list)) else True
    return _config_has(value, rest)


def _resolved_config_supplies_read(config: Mapping[str, Any], dotted: str) -> bool:
    """Return whether the resolved config supplies this adapter read.

    Some fields are derived at materialize time and never appear in a shipped
    template.
    """
    if dotted == CONDITIONAL_SELECTION_GATE:
        targets = config.get("targets")
        if not isinstance(targets, list):
            return False
        # The ranker skips the primary target before reading selection_gate.
        # Requiring that field on the primary creates a false contract failure
        # for every otherwise complete multi-target campaign.
        return all(
            isinstance(target, Mapping)
            and (
                target.get("role") == "primary"
                or isinstance(target.get("selection_gate"), Mapping)
            )
            for target in targets
        )
    return _config_has(config, dotted)


def _config_read_is_reachable(config: Mapping[str, Any], dotted: str) -> bool:
    """Return whether a conditional config read can execute for this campaign.

    ``lane.validate_campaign`` requires ``selection_gate`` only for targets
    whose role is not ``primary``. The ranker skips the primary target before
    it indexes that field. An empty target list and a primary-only list both
    make this read unreachable.

    ``novelty_filter`` requires ``filters.metric_sources`` only when a novelty
    check is enabled. A profile that disables all four was refused at stage 1
    over a field nothing would have read.
    """
    if dotted in CONDITIONAL_CONTIG_KEYS:
        return _routes_contigs_through_the_campaign(config)
    if dotted == CONDITIONAL_METRIC_SOURCES or dotted.startswith(
        CONDITIONAL_METRIC_SOURCES + "."
    ):
        # The adapter reads the mapping and then a templated key inside it. Both
        # reads sit past the same guard, so both are unreachable together.
        return _any_novelty_check_is_enabled(config)
    if dotted != CONDITIONAL_SELECTION_GATE:
        return True
    targets = config.get("targets")
    if not isinstance(targets, list):
        return True
    return any(
        not isinstance(target, Mapping) or target.get("role") != "primary"
        for target in targets
    )


def _routes_contigs_through_the_campaign(config: Mapping[str, Any]) -> bool:
    """Return whether any adapter asks the campaign for its contig value.

    generator_preflight resolves --contigs from the argv template first and only
    falls back to the campaign keys when that value is the {{contigs}} or
    {{generator_contigs}} token. An unreadable adapter list returns True, because
    an audit that cannot tell has to keep the finding.
    """
    adapters = config.get("adapters")
    if not isinstance(adapters, list):
        return True
    for adapter in adapters:
        if not isinstance(adapter, Mapping):
            continue
        command = adapter.get("command_argv_template")
        if not isinstance(command, list):
            continue
        for index, token in enumerate(command[:-1]):
            if token != CONTIG_ARGV_OPTION:
                continue
            value = command[index + 1]
            if not isinstance(value, str):
                continue
            match = CONTIG_ARGV_TOKEN_RE.fullmatch(value)
            if match is not None and match.group(1) in CONTIG_CAMPAIGN_TOKENS:
                return True
    return False


def _any_novelty_check_is_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether the novelty stage still runs a check for this campaign.

    Every check the resolved config declares carries the stage that owns it.
    A check named in ``filters.disabled_checks`` never reaches the adapter, so
    a novelty stage whose checks are all disabled reads no metric source.
    """
    filters = config.get("filters")
    if not isinstance(filters, Mapping):
        return True
    contracts = filters.get("contracts")
    if not isinstance(contracts, list):
        return True
    disabled_raw = filters.get("disabled_checks")
    disabled = set(disabled_raw) if isinstance(disabled_raw, list) else set()
    novelty = [
        contract
        for contract in contracts
        if isinstance(contract, Mapping) and contract.get("stage_id") == "filter-novelty"
    ]
    if not novelty:
        return True
    return any(contract.get("filter_id") not in disabled for contract in novelty)


def _configured_stage_round(config: Mapping[str, Any], stage_id: str) -> int | None:
    """Return the round passed to the filter adapter, when the config resolves it."""
    stages = config.get("stages")
    if isinstance(stages, list):
        for item in stages:
            if not isinstance(item, Mapping) or item.get("stage_id") != stage_id:
                continue
            value = item.get("optimization_round", 0)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
            if value is None:
                return 0
            return None
    match = re.fullmatch(r"optimization-filter-(?:integrity|novelty)-round-(\d+)", stage_id)
    if match is not None:
        return int(match.group(1))
    if stage_id in {"filter-integrity", "filter-novelty"}:
        return 0
    return None


def _round_path_is_unreachable(
    config: Mapping[str, Any], stage_id: str, use: PathUse
) -> bool:
    """Return whether a guarded optimization-round fallback has round zero."""
    return (
        use.guarded
        and use.relative.startswith(OPTIMIZATION_ROUND_PATH_PREFIX)
        and _configured_stage_round(config, stage_id) == 0
    )


def _metric_source_fallback_is_unreachable(
    config: Mapping[str, Any], dotted: str, declared_artifacts: set[str]
) -> bool:
    """Return whether a metric source has a valid declared-artifact route.

    ``novelty_filter.metric_source_paths`` takes the ``produced_by`` branch in
    strict plan mode and continues before it reads the source's fallback path.
    """
    parts = dotted.split(".")
    if len(parts) != 4 or parts[:2] != ["filters", "metric_sources"] or parts[-1] != "path":
        return False
    sources = config.get("filters")
    if not isinstance(sources, Mapping):
        return False
    sources = sources.get("metric_sources")
    if not isinstance(sources, Mapping):
        return False
    source = sources.get(parts[2])
    if not isinstance(source, Mapping):
        return False
    if source.get("kind") == "target-chain-sliding-window":
        return True
    produced_by = source.get("produced_by")
    return isinstance(produced_by, str) and produced_by in declared_artifacts


def _check_bundled_inputs(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: AuditReport,
) -> None:
    """Report a bundled input the plan names at a path that is not there.

    The executor rewrites source paths to bundle-relative ones when it freezes a
    run. A rewrite that misses one key leaves the plan naming a directory that
    never existed, and the stage that reads it fails on its first call.

    The check needs the bundle in front of it. Run away from the run root, every
    absolute path in the plan is absent for a reason that says nothing about the
    plan, so the check reports that it did not run rather than reporting sixteen
    missing files.
    """
    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    bundle = runtime.get("bundle_path")
    if not isinstance(bundle, str) or not Path(bundle).is_dir():
        report.skipped.append(
            f"bundled inputs: runtime.bundle_path is not a directory here ({bundle}), "
            f"so no path in the plan was checked for existence"
        )
        return
    context = plan.get("context") if isinstance(plan.get("context"), Mapping) else {}
    for key, value in sorted(context.items()):
        if not isinstance(value, str) or not value.startswith("/"):
            continue
        if Path(value).exists():
            continue
        report.add(
            Finding(
                stage="<bundle>",
                adapter="<plan context>",
                location=f"run-plan.json context.{key}",
                field=f"context.{key}",
                problem=f"the plan hands every stage this path and it does not exist: {value}",
                fix="rewrite this key to the bundled copy, the way the other context paths point into run-bundle/inputs",
                severity=RUN_FAILING,
            )
        )
    for index, target in enumerate(config.get("targets", []) or []):
        if not isinstance(target, Mapping):
            continue
        for holder, prefix in ((target, f"targets[{index}]"), (target.get("site") or {}, f"targets[{index}].site")):
            if not isinstance(holder, Mapping):
                continue
            for key, value in sorted(holder.items()):
                if not key.endswith("_path") or not isinstance(value, str) or not value.startswith("/"):
                    continue
                if Path(value).exists():
                    continue
                runtime_key = key.startswith("runtime_")
                report.add(
                    Finding(
                        stage="<bundle>",
                        adapter="<resolved config>",
                        location=f"config.resolved.json {prefix}.{key}",
                        field=f"{prefix}.{key}",
                        problem=f"the config names this file and it does not exist: {value}",
                        fix=(
                            "copy the file into run-bundle/inputs and point this key at the copy"
                            if runtime_key
                            else "this is the pre-bundle source path, so confirm the volume is mounted or drop the key"
                        ),
                        severity=RUN_FAILING if runtime_key else SMELL,
                    )
                )


# --- check two: contract against emission ------------------------------------


def check_contract_against_emission(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    report: AuditReport,
    *,
    package_root: Path,
) -> None:
    """Report a declared output the emitting adapter does not produce.

    Both instances of this mechanism fire after the predictor returns, so an
    unfixed one buys GPU work and discards it. That is why these are
    run-failing even though nothing crashes until the parse step.
    """
    bound = _bound_adapters(plan)
    emitted_cache: dict[str, EmissionFacts] = {}

    for stage in _ordered_stages(plan):
        stage_id = str(stage.get("stage_id"))
        adapter_id = str(stage.get("adapter_id"))
        adapter = bound.get(adapter_id)
        if adapter is None:
            continue
        # A stage's rows are written by whichever half of the adapter owns the
        # manifest. Some adapters build every row in `run` and hand the parser a
        # file to count; others build them in `parse`. Reading only one half
        # reported every field of two stages as unemitted, so both are read.
        entries = [
            entry
            for entry in (
                _entry_module(adapter.get("command_argv_template") or []),
                _entry_module(adapter.get("parser_argv_template") or []),
            )
            if entry is not None
        ]
        if not entries:
            continue
        if adapter_id not in emitted_cache:
            keys: set[str] = set()
            mentioned: set[str] = set()
            passthrough: list[str] = []
            offpath: list[str] = []
            for entry in entries:
                # ``python -m claude_binder`` executes the package CLI in
                # lane.py. Path analysis already applies this translation via
                # _analysis_entry; emission analysis must inspect the same
                # selected program instead of package __init__.py.
                analysis_entry = _analysis_entry(entry)
                for source in module_closure(
                    analysis_entry, package_root, depth=None
                ).values():
                    facts = collect_emission_facts(source)
                    keys |= facts.keys
                    mentioned |= facts.mentioned
                    passthrough.extend(facts.passthrough)
                    offpath.extend(facts.offpath_imports)
            emitted_cache[adapter_id] = EmissionFacts(
                frozenset(keys),
                frozenset(mentioned),
                tuple(passthrough),
                tuple(sorted(set(offpath))),
            )
        facts = emitted_cache[adapter_id]

        for raw in stage.get("outputs", []) or []:
            if not isinstance(raw, Mapping):
                continue
            artifact_id = str(raw.get("artifact_id"))
            kind = str(raw.get("kind", ""))
            fields = [str(value) for value in (raw.get("required_fields") or []) if isinstance(value, str)]
            template = raw.get("path_template") if isinstance(raw.get("path_template"), str) else ""

            if fields and kind in FIELD_CHECKED_KINDS and facts.mentioned:
                for name in fields:
                    if name in facts.mentioned:
                        continue
                    copied = facts.passthrough[0] if facts.passthrough else ""
                    offpath_note = facts.offpath_imports[0] if facts.offpath_imports else ""
                    caveat = ""
                    if offpath_note:
                        caveat = f", and {offpath_note}, so the row builder is not all readable here"
                    elif copied:
                        caveat = f", though {copied}, so an upstream row could carry it"
                    report.add(
                        Finding(
                            stage=stage_id,
                            adapter=adapter_id,
                            location=f"run-plan.json stages[{stage_id}].outputs[{artifact_id}].required_fields",
                            field=name,
                            problem=(
                                f"{stage_id} declares {name} on every {artifact_id} record and the "
                                f"{adapter_id} source never names {name} anywhere" + caveat
                            ),
                            fix=(
                                f"have {adapter_id} write {name} on every row, or drop {name} from "
                                f"{artifact_id}.required_fields"
                            ),
                            severity=RUN_FAILING if (facts.complete and not copied) else SMELL,
                            confidence=UNCERTAIN,
                        )
                    )

            if fields and kind not in FIELD_CHECKED_KINDS:
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=adapter_id,
                        location=f"run-plan.json stages[{stage_id}].outputs[{artifact_id}]",
                        field=f"{artifact_id}.required_fields",
                        problem=(
                            f"required_fields is declared on a {kind} output and "
                            f"lane.validate_artifact checks fields only for json and jsonl, "
                            f"so this contract is never enforced"
                        ),
                        fix=f"change the kind to jsonl, or drop required_fields from {artifact_id}",
                        severity=SMELL,
                    )
                )

            suffixes = KIND_SUFFIXES.get(kind)
            if suffixes and template:
                actual = PurePosixPath(template).suffix.lower()
                if actual and actual not in suffixes:
                    report.add(
                        Finding(
                            stage=stage_id,
                            adapter=adapter_id,
                            location=f"run-plan.json stages[{stage_id}].outputs[{artifact_id}].path_template",
                            field=f"{artifact_id}.kind",
                            problem=(
                                f"the declared kind is {kind} and the path_template ends in {actual}, "
                                f"which lane.validate_artifact will parse as {kind}"
                            ),
                            fix=f"set kind to the one matching {actual}, or correct the path_template",
                            severity=RUN_FAILING,
                        )
                    )

            if kind in STRUCTURE_KINDS and raw.get("records_per_count") is not None:
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=adapter_id,
                        location=f"run-plan.json stages[{stage_id}].outputs[{artifact_id}].records_per_count",
                        field=f"{artifact_id}.records_per_count",
                        problem=(
                            f"a record of a {kind} artifact is whatever lane.pdb_pose_count returns, "
                            f"which is one per MODEL block or one for a file with no MODEL delimiters, "
                            f"and nothing checks that {adapter_id} writes that many"
                        ),
                        fix=(
                            f"confirm {adapter_id} writes {raw.get('records_per_count')} pose(s) per "
                            f"invocation, and record where you confirmed it"
                        ),
                        severity=SMELL,
                        confidence=UNCERTAIN,
                    )
                )

    _check_field_provenance(plan, report, emitted_cache)
    _check_unbound_adapters(plan, report)


def _check_field_provenance(
    plan: Mapping[str, Any],
    report: AuditReport,
    facts_by_adapter: Mapping[str, EmissionFacts],
) -> None:
    """Report a row field whose only in-plan producer runs later than the row.

    This is the plan disagreeing with itself, so it needs no adapter source and
    it holds whatever the adapters do next. Both ESMFold2-Fast stages require
    ``msa_path`` and ``msa_sha256``, the only stage in the plan that produces an
    alignment is ``stage-msa``, and the plan orders ``stage-msa`` after both of
    them. A stage cannot carry a value nothing has produced yet.

    A stage running the same adapter is not a producer. Two cofold stages
    declaring the same field are making the same claim twice, and counting one
    as the other's source hides the second instance.
    """
    stages = _ordered_stages(plan)
    position = {str(stage.get("stage_id")): index for index, stage in enumerate(stages)}
    declarers: dict[str, list[tuple[str, str, int]]] = {}
    for index, stage in enumerate(stages):
        stage_id = str(stage.get("stage_id"))
        adapter_id = str(stage.get("adapter_id"))
        for raw in stage.get("outputs", []) or []:
            if not isinstance(raw, Mapping):
                continue
            for name in raw.get("required_fields") or []:
                if isinstance(name, str):
                    declarers.setdefault(name, []).append((stage_id, adapter_id, index))

    for index, stage in enumerate(stages):
        stage_id = str(stage.get("stage_id"))
        adapter_id = str(stage.get("adapter_id"))
        for raw in stage.get("outputs", []) or []:
            if not isinstance(raw, Mapping):
                continue
            artifact_id = str(raw.get("artifact_id"))
            for name in raw.get("required_fields") or []:
                if not isinstance(name, str):
                    continue
                facts = facts_by_adapter.get(adapter_id)
                if facts is not None and name in facts.keys:
                    # The adapter writes this key itself, so the stage
                    # originates the field rather than sourcing it. Every
                    # generator declares candidate_id and nothing upstream
                    # supplies it, and that is correct.
                    continue
                others = [
                    item for item in declarers.get(name, []) if item[1] != adapter_id
                ]
                if not others:
                    continue
                if any(item[2] < index for item in others):
                    continue
                earliest = min(others, key=lambda item: item[2])
                report.add(
                    Finding(
                        stage=stage_id,
                        adapter=adapter_id,
                        location=f"run-plan.json stages[{stage_id}].outputs[{artifact_id}].required_fields",
                        field=name,
                        problem=(
                            f"{stage_id} runs at position {index} and requires {name} on every "
                            f"{artifact_id} record, and the only other stage that declares {name} is "
                            f"{earliest[0]} at position {earliest[2]}"
                        ),
                        fix=(
                            f"move {earliest[0]} before {stage_id} in ordered_stage_ids and add it to "
                            f"{stage_id}.depends_on, or drop {name} from {artifact_id}.required_fields "
                            f"because this arm does not produce it"
                        ),
                        severity=RUN_FAILING,
                    )
                )


def _check_unbound_adapters(plan: Mapping[str, Any], report: AuditReport) -> None:
    """Report an adapter the plan registers and no stage runs.

    An unbound adapter is the usual reason a path has no producer. It also
    means the runtime check validates a tool the run will never call.
    """
    bound = set(_bound_adapters(plan))
    for adapter in plan.get("adapters", []) or []:
        if not isinstance(adapter, Mapping):
            continue
        adapter_id = str(adapter.get("adapter_id"))
        if adapter_id in bound:
            continue
        produced = ", ".join(str(value) for value in (adapter.get("produced_artifacts") or [])) or "nothing"
        report.add(
            Finding(
                stage="<none>",
                adapter=adapter_id,
                location=f"run-plan.json adapters[{adapter_id}]",
                field=f"{adapter_id}.produced_artifacts",
                problem=f"the plan registers this adapter, no stage runs it, and it is the declared producer of {produced}",
                fix=f"add a stage bound to {adapter_id}, or drop it from the plan's adapters",
                severity=SMELL,
            )
        )


# --- entry points ------------------------------------------------------------


def audit_run(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    package_root: Path | None = None,
    template_root: Path | None = None,
) -> AuditReport:
    """Return every disagreement between this plan and the code it will run.

    This runs the same checks as preflight_plan, because a scientist runs
    contract-audit before authorizing paid work and a pre-authorization check
    that misses the gate is worse than no check. It once ran five of the seven,
    so a plan could audit clean and then be refused by preflight for a config
    key no shipped template sets.
    """
    root = _package_root(package_root)
    report = AuditReport()
    check_declared_inputs(plan, report)
    check_adapter_argument_templates(plan, report, package_root=root)
    check_undeclared_reads(config, plan, report, package_root=root)
    check_executor_artifact_reads(plan, report, package_root=root)
    check_filter_lineage_contract_closure(config, plan, report)
    check_contract_against_emission(config, plan, report, package_root=root)
    check_profile_config_coverage(
        plan,
        report,
        config=config,
        package_root=root,
        template_root=(template_root or root / "data" / "templates").resolve(),
    )
    return report


def audit_paths(
    config_path: Path,
    plan_path: Path,
    *,
    package_root: Path | None = None,
    template_root: Path | None = None,
) -> AuditReport:
    """Read a resolved config and a run plan off disk and audit them."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    return audit_run(
        config, plan, package_root=package_root, template_root=template_root
    )


def _preflight_order(plan: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(stage.get("stage_id")): index
        for index, stage in enumerate(_ordered_stages(plan))
    }


def preflight_plan(
    plan: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
    *,
    package_root: Path | None = None,
    template_root: Path | None = None,
) -> AuditReport:
    """Run the strict pre-spend audit against one materialized plan."""
    root = _package_root(package_root)
    report = AuditReport()
    check_declared_inputs(plan, report)
    if config is not None:
        check_adapter_argument_templates(plan, report, package_root=root)
        check_undeclared_reads(config, plan, report, package_root=root)
        check_executor_artifact_reads(plan, report, package_root=root)
        check_filter_lineage_contract_closure(config, plan, report)
        check_contract_against_emission(config, plan, report, package_root=root)
    else:
        report.skipped.append("resolved config: no config.resolved.json was supplied beside the plan")
    check_profile_config_coverage(
        plan,
        report,
        config=config,
        package_root=root,
        template_root=(template_root or root / "data" / "templates").resolve(),
    )
    order = _preflight_order(plan)
    report.findings.sort(
        key=lambda item: (
            order.get(item.stage, len(order)),
            0 if item.severity == RUN_FAILING else 1,
            item.adapter,
            item.field,
            item.location,
        )
    )
    return report


def preflight_paths(
    plan_path: Path,
    config_path: Path | None = None,
    *,
    package_root: Path | None = None,
    template_root: Path | None = None,
) -> AuditReport:
    """Read a materialized plan and its adjacent resolved config, when present."""
    plan_path = Path(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    resolved_config = Path(config_path) if config_path is not None else plan_path.parent / "config.resolved.json"
    config: Mapping[str, Any] | None = None
    if resolved_config.is_file():
        config = json.loads(resolved_config.read_text(encoding="utf-8"))
    return preflight_plan(
        plan,
        config,
        package_root=package_root,
        template_root=template_root,
    )


def preflight_text(report: AuditReport) -> str:
    """Render strict findings in the order the preflight checked them."""
    if not report.findings:
        return "preflight: no findings"
    lines = [f"preflight: {len(report.findings)} findings"]
    for finding in report.findings:
        lines.extend(("", finding.text()))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-binder contract-audit",
        description="Compare a run plan against the adapter code it will run.",
    )
    parser.add_argument("--config", type=Path, required=True, help="resolved campaign config")
    parser.add_argument("--plan", type=Path, required=True, help="materialized run plan")
    parser.add_argument("--package-root", type=Path, default=None, help="adapter source root to read")
    parser.add_argument("--json", action="store_true", help="write the report as JSON")
    parser.add_argument(
        "--include-smells",
        action="store_true",
        help="exit non-zero on a smell as well as on a run-failing finding",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_paths(args.config, args.plan, package_root=args.package_root)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(report.text())
    if args.include_smells:
        return 0 if not report.findings else 1
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
