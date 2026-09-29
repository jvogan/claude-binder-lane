"""Entry point for ``python -m claude_binder``.

The call is guarded so that importing this module does not run the CLI. An
unguarded call exits during import, which means a test that imports every
module in the built artifact cannot cover this one, and that test is the only
thing standing between a missing module and a shipped skill that cannot start.
``python -m`` still works, because it sets ``__name__`` to ``"__main__"``.
"""

from .lane import cli

if __name__ == "__main__":
    raise SystemExit(cli())
