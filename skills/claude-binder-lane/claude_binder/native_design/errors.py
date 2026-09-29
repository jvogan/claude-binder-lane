"""Stable native-design refusal codes shared by validation and runtime code."""

from __future__ import annotations


FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE = "DESIGN_CHECKPOINT_UNAVAILABLE"
FAILURE_DESIGN_AUTH_REJECTED = "DESIGN_AUTH_REJECTED"
FAILURE_DESIGN_EGRESS_BLOCKED = "DESIGN_EGRESS_BLOCKED"
FAILURE_EPITOPE_OUT_OF_RANGE = "EPITOPE_OUT_OF_RANGE"
FAILURE_UNAIMED_NOT_ACKNOWLEDGED = "UNAIMED_NOT_ACKNOWLEDGED"
FAILURE_NO_MUTABLE_POSITIONS = "NO_MUTABLE_POSITIONS"
FAILURE_ALPHABET_INVALID = "ALPHABET_INVALID"
FAILURE_PATTERN_COLLISION = "PATTERN_COLLISION"
FAILURE_CUDA_OOM_EXHAUSTED = "CUDA_OOM_EXHAUSTED"
FAILURE_CRITIC_FOLD_FAILED = "CRITIC_FOLD_FAILED"

FAILURE_CODES = frozenset(
    {
        FAILURE_DESIGN_CHECKPOINT_UNAVAILABLE,
        FAILURE_DESIGN_AUTH_REJECTED,
        FAILURE_DESIGN_EGRESS_BLOCKED,
        FAILURE_EPITOPE_OUT_OF_RANGE,
        FAILURE_UNAIMED_NOT_ACKNOWLEDGED,
        FAILURE_NO_MUTABLE_POSITIONS,
        FAILURE_ALPHABET_INVALID,
        FAILURE_PATTERN_COLLISION,
        FAILURE_CUDA_OOM_EXHAUSTED,
        FAILURE_CRITIC_FOLD_FAILED,
    }
)


class DesignRefusal(RuntimeError):
    """A stable, actionable condition that stops native design."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown native-design failure code: {code}")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")
