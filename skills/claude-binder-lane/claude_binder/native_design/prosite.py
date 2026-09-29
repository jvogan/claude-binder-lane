"""CPU-only PROSITE parsing and prompt placement for native design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .constants import (
    AMBIGUOUS_AMINO_ACIDS,
    MUTABLE_TOKEN,
    PROSITE_COUNT_RE,
    STANDARD_AMINO_ACID_SET,
)


@dataclass(frozen=True)
class ParsedPattern:
    """A parsed PROSITE signature."""

    elements: tuple[tuple[frozenset[str], int, int], ...]
    n_anchor: bool
    c_anchor: bool
    raw: str

    @property
    def min_length(self) -> int:
        return sum(low for _, low, _ in self.elements)

    @property
    def max_length(self) -> int:
        return sum(high for _, _, high in self.elements)


def _expand_letters(value: str) -> frozenset[str]:
    result: set[str] = set()
    for character in value.upper():
        if character in AMBIGUOUS_AMINO_ACIDS:
            result.update(AMBIGUOUS_AMINO_ACIDS[character])
        elif character in STANDARD_AMINO_ACID_SET:
            result.add(character)
    return frozenset(result)


def _parse_element(value: str) -> tuple[frozenset[str], int, int]:
    matched = PROSITE_COUNT_RE.search(value)
    if matched is None:
        lower = upper = 1
        body = value
    else:
        lower = int(matched.group(1))
        upper = int(matched.group(2)) if matched.group(2) is not None else lower
        body = value[: matched.start()]
    body = body.strip()
    if body in {"x", "X"}:
        allowed = STANDARD_AMINO_ACID_SET
    elif body.startswith("[") and body.endswith("]"):
        allowed = _expand_letters(body[1:-1])
    elif body.startswith("{") and body.endswith("}"):
        allowed = frozenset(STANDARD_AMINO_ACID_SET - _expand_letters(body[1:-1]))
    elif len(body) == 1 and (
        body.upper() in STANDARD_AMINO_ACID_SET
        or body.upper() in AMBIGUOUS_AMINO_ACIDS
    ):
        allowed = _expand_letters(body)
    else:
        raise ValueError(f"unparseable PROSITE element: {value!r}")
    if not allowed:
        raise ValueError(f"PROSITE element allows no standard amino acid: {value!r}")
    if lower < 1 or upper < lower:
        raise ValueError(f"invalid PROSITE repeat count in element: {value!r}")
    return allowed, lower, upper


def parse_prosite(pattern: str) -> ParsedPattern:
    """Parse a PROSITE pattern without importing a model library."""

    raw = pattern.strip()
    body = raw.rstrip(".").strip()
    if not body:
        raise ValueError("empty PROSITE pattern")
    parts = [part for part in body.split("-") if part]
    n_anchor = bool(parts and parts[0].startswith("<"))
    c_anchor = bool(parts and parts[-1].endswith(">"))
    if n_anchor:
        parts[0] = parts[0][1:]
    if c_anchor:
        parts[-1] = parts[-1][:-1]
    parts = [part for part in parts if part]
    if not parts:
        raise ValueError(f"PROSITE pattern has no elements: {pattern!r}")
    return ParsedPattern(
        elements=tuple(_parse_element(part) for part in parts),
        n_anchor=n_anchor,
        c_anchor=c_anchor,
        raw=raw,
    )


def realize(
    pattern: ParsedPattern,
    *,
    gap: str = "min",
    overrides: Mapping[int, int] | None = None,
) -> list[frozenset[str]]:
    """Resolve variable repeats into one allowed set per binder position."""

    if gap not in {"min", "max"}:
        raise ValueError("pattern gap must be 'min' or 'max'")
    result: list[frozenset[str]] = []
    for index, (allowed, lower, upper) in enumerate(pattern.elements):
        if overrides is not None and index in overrides:
            count = int(overrides[index])
            if not lower <= count <= upper:
                raise ValueError(
                    f"pattern repeat override {count} at element {index} is outside [{lower}, {upper}]"
                )
        elif lower == upper:
            count = lower
        else:
            count = lower if gap == "min" else upper
        result.extend([allowed] * count)
    return result


def to_regex(pattern: ParsedPattern) -> str:
    """Return the Python regular expression for a parsed signature."""

    chunks = ["^"] if pattern.n_anchor else []
    for allowed, lower, upper in pattern.elements:
        body = "." if allowed == STANDARD_AMINO_ACID_SET else f"[{''.join(sorted(allowed))}]"
        repeat = "" if lower == upper == 1 else (f"{{{lower}}}" if lower == upper else f"{{{lower},{upper}}}")
        chunks.append(body + repeat)
    if pattern.c_anchor:
        chunks.append("$")
    return "".join(chunks)


def place(
    pattern: ParsedPattern,
    length: int,
    *,
    gap: str = "min",
    anchor: str | None = None,
    start: int | None = None,
    overrides: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Place a realized signature in a binder prompt."""

    if length < 1:
        raise ValueError("binder length must be positive")
    realized = realize(pattern, gap=gap, overrides=overrides)
    motif_length = len(realized)
    if motif_length > length:
        raise ValueError(f"pattern length {motif_length} exceeds binder length {length}")
    if start is not None:
        chosen = start
    elif pattern.n_anchor or anchor == "n":
        chosen = 0
    elif pattern.c_anchor or anchor == "c":
        chosen = length - motif_length
    elif anchor in {None, "center"}:
        chosen = (length - motif_length) // 2
    else:
        raise ValueError(f"unknown pattern anchor: {anchor!r}")
    if chosen < 0 or chosen + motif_length > length:
        raise ValueError(f"pattern start {chosen} and length {motif_length} exceed binder length {length}")
    prompt = [MUTABLE_TOKEN] * length
    position_allowed: dict[int, frozenset[str]] = {}
    motif: list[dict[str, Any]] = []
    for offset, allowed in enumerate(realized):
        position = chosen + offset
        if len(allowed) == 1:
            prompt[position] = next(iter(allowed))
            kind = "fixed"
        elif allowed == STANDARD_AMINO_ACID_SET:
            kind = "any"
        else:
            position_allowed[position] = allowed
            kind = "restricted"
        motif.append({
            "position_0based": position,
            "position_1based": position + 1,
            "allowed": "".join(sorted(allowed)),
            "kind": kind,
        })
    return {
        "prompt": "".join(prompt),
        "position_allowed": position_allowed,
        "motif": motif,
        "pattern": pattern.raw,
        "regex": to_regex(pattern),
        "start_0based": chosen,
        "motif_length": motif_length,
        "n_anchor": pattern.n_anchor,
        "c_anchor": pattern.c_anchor,
    }


def build_prosite_prompt(
    pattern: str,
    length: int,
    *,
    gap: str = "min",
    anchor: str | None = None,
    start: int | None = None,
    overrides: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Parse and place one PROSITE signature."""

    return place(parse_prosite(pattern), length, gap=gap, anchor=anchor, start=start, overrides=overrides)


def overlay_signatures(prompt: str, layouts: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Overlay placed signatures and refuse every incompatible fixed residue."""

    characters = list(prompt)
    restrictions: dict[int, frozenset[str]] = {}
    motif: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for layout in layouts:
        for entry in layout.get("motif", []):
            position = int(entry["position_0based"])
            allowed = frozenset(str(entry["allowed"]))
            existing = characters[position]
            if existing != MUTABLE_TOKEN:
                if existing not in allowed:
                    collisions.append({"position_1based": position + 1, "existing": existing, "allowed": "".join(sorted(allowed))})
                continue
            if entry["kind"] == "fixed":
                characters[position] = next(iter(allowed))
            elif entry["kind"] == "restricted":
                restrictions[position] = allowed
        motif.extend(layout.get("motif", []))
    if collisions:
        raise ValueError(f"PROSITE pattern collides with fixed residues: {collisions}")
    return {"prompt": "".join(characters), "position_allowed": restrictions, "motif": motif}
