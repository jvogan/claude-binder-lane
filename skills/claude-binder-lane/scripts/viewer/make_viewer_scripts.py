#!/usr/bin/env python3
"""Run the canonical viewer-script builder from a checkout or an installed skill."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve()
SKILL_ROOT = HERE.parents[2]
REPOSITORY_SRC = SKILL_ROOT.parents[1] / "src"

for import_root in (SKILL_ROOT, REPOSITORY_SRC):
    if (import_root / "claude_binder").is_dir():
        sys.path.insert(0, str(import_root))
        break
else:
    raise SystemExit(
        "claude_binder package not found. Reinstall this skill; it does not "
        "ship the package this script runs.")

from claude_binder.data.helpers.viewer.make_viewer_scripts import main


if __name__ == "__main__":
    raise SystemExit(main())
