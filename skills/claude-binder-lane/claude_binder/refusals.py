"""Build actionable refusals and map them to process exit codes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class ExitCode(IntEnum):
    """Process results that callers can act on."""

    VERIFIED = 0
    FAILURE = 1
    PREFLIGHT_REFUSAL = 2
    POST_SPEND_REFUSAL = 3


def exit_code_for_result(
    *,
    verified: bool,
    refused: bool = False,
    paid_stage_completed: bool = False,
) -> int:
    """Return the status code for one completed check or execution result.

    A caller may set ``verified`` only after promised outputs exist, open, and
    pass their result gate. A refusal takes precedence over ``verified`` so a
    caller cannot return success for a refusal.
    """
    if refused:
        if paid_stage_completed:
            return int(ExitCode.POST_SPEND_REFUSAL)
        return int(ExitCode.PREFLIGHT_REFUSAL)
    if verified:
        return int(ExitCode.VERIFIED)
    return int(ExitCode.FAILURE)


@dataclass(frozen=True)
class Refusal:
    """One refusal with its operative condition and a repair owner."""

    cause: str
    expected: str
    expected_source: str
    found: str
    found_source: str
    scope: str
    action: str
    escalation: str
    paid_stage_completed: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "cause",
            "expected",
            "expected_source",
            "found",
            "found_source",
            "scope",
            "action",
            "escalation",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"refusal {field_name} must be a non-empty string")

    @property
    def exit_code(self) -> int:
        """Return the nonzero status code required for this refusal."""
        return exit_code_for_result(
            verified=False,
            refused=True,
            paid_stage_completed=self.paid_stage_completed,
        )

    def text(self) -> str:
        """Render the human-readable refusal for standard error."""
        return "\n".join(
            (
                f"Stopped: {self.cause}",
                f"Expected: {self.expected} Source: {self.expected_source}",
                f"Found: {self.found} Source: {self.found_source}",
                f"Scope: {self.scope}",
                f"Action: {self.action}",
                f"Escalation: {self.escalation}",
            )
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable refusal record for result JSON."""
        return {
            "cause": self.cause,
            "expected": self.expected,
            "expected_source": self.expected_source,
            "found": self.found,
            "found_source": self.found_source,
            "scope": self.scope,
            "action": self.action,
            "escalation": self.escalation,
            "paid_stage_completed": self.paid_stage_completed,
            "exit_code": self.exit_code,
            "text": self.text(),
        }
