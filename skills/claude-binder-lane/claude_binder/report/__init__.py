"""Report generation package for computational protein binder design campaigns."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Import the submodules first. `build_report` and `contract_report` are both a
# submodule name and a legacy callable name, and the import system binds the
# submodule onto this package the first time it is loaded. Loading them here
# means the legacy names copied in below win permanently, instead of the
# binding depending on whether some other module imported the submodule first.
from . import build_report as _build_report_submodule  # noqa: F401
from . import contract_report as _contract_report_submodule  # noqa: F401

_report_py = Path(__file__).resolve().parent.parent / "report.py"
if _report_py.is_file():
    _spec = importlib.util.spec_from_file_location("claude_binder._report_legacy", _report_py)
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _mod.__package__ = "claude_binder"
        sys.modules["claude_binder._report_legacy"] = _mod
        _spec.loader.exec_module(_mod)
        for _name in dir(_mod):
            if not _name.startswith("__"):
                globals()[_name] = getattr(_mod, _name)

