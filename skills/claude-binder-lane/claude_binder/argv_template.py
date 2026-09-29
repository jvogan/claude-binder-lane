"""Render argv templates with the lane's declared placeholder values."""

from __future__ import annotations

import re
from typing import Any, Sequence


TOKEN_RE = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


def render_tokens(template: str, context: dict[str, Any]) -> str:
    """Render one argv token from a fully resolved execution context."""
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise ValueError(f"command token has no value: {key}")
        return str(context[key])

    rendered = TOKEN_RE.sub(replace, template)
    if "{{" in rendered or "}}" in rendered:
        raise ValueError("command contains an unresolved or malformed token")
    return rendered


def render_argv(template: Sequence[str], context: dict[str, Any]) -> list[str]:
    """Render every token in one argv template."""
    return [render_tokens(item, context) for item in template]
