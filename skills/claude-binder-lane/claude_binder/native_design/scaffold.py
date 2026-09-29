"""CPU scaffold prompt helpers retained from the sister design package.

The binder adapter does not route through these experimental single-chain
helpers. They remain available for fixed-motif work without importing Torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import MUTABLE_TOKEN, STANDARD_AMINO_ACIDS


@dataclass(frozen=True)
class ScaffoldPrompt:
    bin_id: str
    display_name: str
    design_index: int
    seed: int
    length: int
    prompt: str
    fixed_positions: list[dict[str, Any]]
    allowed_amino_acids: str
    design_mode: str = "compact_shell"
    barrel_segment_count: int = 8
    barrel_strand_segments: list[dict[str, Any]] = field(default_factory=list)
    barrel_pair_targets: list[list[int]] = field(default_factory=list)
    chromophore_geometry: bool = False


NATIVE_EGFP_LENGTH = 228


def _parse_residue_label(label: str) -> tuple[str, int]:
    if len(label) < 2 or label[0].upper() not in STANDARD_AMINO_ACIDS:
        raise ValueError(f"unsupported fixed residue label: {label}")
    return label[0].upper(), int(label[1:])


def _claim_position(prompt: list[str], preferred: int, residue: str, source_label: str, fixed: list[dict[str, Any]]) -> None:
    for offset in range(len(prompt)):
        for position in (preferred - offset, preferred + offset):
            if 1 <= position <= len(prompt) and prompt[position - 1] == MUTABLE_TOKEN:
                prompt[position - 1] = residue
                fixed.append({"source_label": source_label, "aa": residue, "position_1based": position, "position_0based": position - 1})
                return
    raise ValueError(f"could not place fixed residue {source_label} in length {len(prompt)}")


def build_chromashell_prompt(
    *,
    bin_id: str,
    display_name: str,
    design_index: int,
    seed: int,
    length: int,
    fixed_residue_labels: list[str],
    allowed_amino_acids: str,
) -> ScaffoldPrompt:
    """Build the sister fixed-chromophore prompt without tensor dependencies."""

    if length < 3:
        raise ValueError("chromashell length must be at least three")
    prompt = [MUTABLE_TOKEN] * length
    fixed: list[dict[str, Any]] = []
    labels = set(fixed_residue_labels)
    if {"T65", "Y66", "G67"}.issubset(labels):
        start = max(1, min(length - 2, round(65 / NATIVE_EGFP_LENGTH * length)))
        for residue, label, position in (("T", "T65", start), ("Y", "Y66", start + 1), ("G", "G67", start + 2)):
            _claim_position(prompt, position, residue, label, fixed)
    for label in fixed_residue_labels:
        if label in {"T65", "Y66", "G67"}:
            continue
        residue, native_position = _parse_residue_label(label)
        _claim_position(prompt, max(1, min(length, round(native_position / NATIVE_EGFP_LENGTH * length))), residue, label, fixed)
    return ScaffoldPrompt(bin_id, display_name, design_index, seed, length, "".join(prompt), sorted(fixed, key=lambda item: item["position_1based"]), allowed_amino_acids)


def build_beta_barrel_chromashell_prompt(**kwargs: Any) -> ScaffoldPrompt:
    """Return a chromashell prompt with the sister barrel metadata shape."""

    segment_count = int(kwargs.pop("barrel_segment_count", 8))
    chromophore_geometry = bool(kwargs.pop("chromophore_geometry", False))
    base = build_chromashell_prompt(**kwargs)
    segment_length = max(3, min(8, round(base.length * 0.045)))
    segments = []
    for index in range(max(4, min(11, segment_count))):
        center = round((index + 0.5) * base.length / segment_count)
        start = max(1, min(base.length - segment_length + 1, center - segment_length // 2))
        positions = list(range(start, start + segment_length))
        segments.append({"strand_index": index, "positions_1based": positions, "start_1based": positions[0], "end_1based": positions[-1]})
    pairs = [[index, (index + 1) % len(segments)] for index in range(len(segments))]
    return ScaffoldPrompt(
        **{**base.__dict__, "design_mode": "beta_barrel_bias", "barrel_segment_count": segment_count, "barrel_strand_segments": segments, "barrel_pair_targets": pairs, "chromophore_geometry": chromophore_geometry}
    )
