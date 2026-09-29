"""CPU representations of native-design sequence and contact constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .constants import MUTABLE_TOKEN, STANDARD_AMINO_ACID_SET
from .errors import DesignRefusal, FAILURE_PATTERN_COLLISION


@dataclass(frozen=True)
class GradientMask:
    prompt: str
    allowed_amino_acids: tuple[frozenset[str], ...]
    mutable_positions: tuple[int, ...]


@dataclass(frozen=True)
class ContactMask:
    chain_mask: tuple[bool, ...]
    binder_mask: tuple[bool, ...]
    aimed: bool
    epitope_indices_0based: tuple[int, ...]


def build_gradient_mask(
    prompt: str,
    *,
    allowed_amino_acids: str,
    position_allowed: Mapping[int, Iterable[str]] | None = None,
) -> GradientMask:
    """Build the CPU mask that the tensor loop converts into gradient masks."""

    global_allowed = frozenset(allowed_amino_acids)
    if not global_allowed or global_allowed - STANDARD_AMINO_ACID_SET:
        raise ValueError("allowed_amino_acids contains no valid standard amino acids")
    values: list[frozenset[str]] = []
    mutable: list[int] = []
    for position, residue in enumerate(prompt):
        if residue == MUTABLE_TOKEN:
            allowed = global_allowed if position_allowed is None or position not in position_allowed else frozenset(position_allowed[position])
            if not allowed or allowed - STANDARD_AMINO_ACID_SET:
                raise ValueError(f"position {position + 1} has an invalid allowed amino-acid set")
            values.append(allowed)
            mutable.append(position)
        elif residue in STANDARD_AMINO_ACID_SET:
            if position_allowed is not None and position in position_allowed:
                allowed = frozenset(position_allowed[position])
                if residue not in allowed:
                    raise DesignRefusal(
                        FAILURE_PATTERN_COLLISION,
                        f"fixed residue {residue} conflicts with its pattern at position {position + 1}",
                    )
            values.append(frozenset({residue}))
        else:
            raise ValueError(f"binder prompt contains invalid residue {residue!r} at position {position + 1}")
    return GradientMask(prompt=prompt, allowed_amino_acids=tuple(values), mutable_positions=tuple(mutable))


def build_inter_contact_masks(
    target_length: int,
    binder_length: int,
    epitope_indices_0based: Iterable[int] | None,
) -> ContactMask:
    """Build the target averaging mask and binder minimization mask."""

    if target_length < 1 or binder_length < 1:
        raise ValueError("target_length and binder_length must be positive")
    total = target_length + binder_length
    binder_mask = tuple(index >= target_length for index in range(total))
    if epitope_indices_0based is None:
        return ContactMask(
            chain_mask=tuple(index < target_length for index in range(total)),
            binder_mask=binder_mask,
            aimed=False,
            epitope_indices_0based=(),
        )
    epitope = tuple(sorted(set(epitope_indices_0based)))
    if not epitope or epitope[0] < 0 or epitope[-1] >= target_length:
        raise ValueError("epitope indices must identify target residues")
    selected = set(epitope)
    return ContactMask(
        chain_mask=tuple(index in selected for index in range(total)),
        binder_mask=binder_mask,
        aimed=True,
        epitope_indices_0based=epitope,
    )
