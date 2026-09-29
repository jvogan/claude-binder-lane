"""Native ESMFold2 design implementation with lazy tensor imports.

Importing this package performs only CPU-safe setup. The GPU runtime imports
Torch, ESMFold2, Transformers, and Biotite inside the run boundary.
"""

from .runtime import RuntimePreparation, prepare_runtime_request, run_native_design

__all__ = (
    "RuntimePreparation",
    "prepare_runtime_request",
    "run_native_design",
)
