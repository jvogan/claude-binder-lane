"""Write live progress narration without mixing it into structured output."""

from __future__ import annotations

import sys
from typing import TextIO


class Progress:
    """Emit one human-readable progress line to a dedicated stream.

    The reporter flushes after every line so an operator sees stage completion
    as soon as the executor observes it. The default stream is stderr because
    the executor reserves stdout for its final machine-readable result.
    """

    def __init__(self, stream: TextIO | None = None, *, enabled: bool = True) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled

    def __call__(self, line: str) -> None:
        if not self.enabled:
            return
        clean_line = line.rstrip("\r\n")
        self.stream.write(clean_line + "\n")
        self.stream.flush()
