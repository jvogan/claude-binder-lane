"""Pinned Modal stack for Anthropic's PXDesign optimization kit.

This preserves the upstream CUDA development image and build-time LayerNorm
extension, with the explicit one-line stock-isolation driver repair recorded
in META. build() only declares resources. HYDRATE verifies all seven weight
and CCD files; CHECK needs a GPU. The upstream mode is never substituted.
Source: pxdesign/environment/Dockerfile in uplifting-biomolecular-modeling at
f4f62fa6592ae4938d49b1757bea0cfeff9f468e.
"""

from __future__ import annotations

import modal


META = {
    "packages": ["torch", "triton", "numpy", "deepspeed", "protenix", "pxdesign", "pxdbench", "pxdesign_opt", "opt_core"],
    "gpu_default": "H100",
    "supersedes": [],
    "egress_domains": ["pxdesign.tos-cn-beijing.volces.com"],
}

_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_KIT_ROOT = "/kit/pxdesign"
_WEIGHTS_MOUNT = "/weights"
_WEIGHTS_VOLUME_NAME = "claude-science-pxdesign-kit-weights"
_JIT_CACHE_ROOT = "/weights/jit"
_PYTHON_ARCHIVE = "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
_PYTHON_DIGEST = "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1"

# The deterministic stock driver unnecessarily imports the activation module.
# Its after-call isolation proof forbids that import. This diff changes only
# the unused import: source/model pins, activation APIs and sampling stay intact.
_STOCK_ISOLATION_PATCH = '''--- a/pxdesign/opt/pxdesign_opt/infer_loop.py
+++ b/pxdesign/opt/pxdesign_opt/infer_loop.py
@@ -126,7 +126,6 @@
         configs.input_json_path = os.path.join(configs.dump_dir, "input_tasks.json")
         with open(configs.input_json_path, "w") as f:
             json.dump(orig_inputs, f, indent=4)
-    from . import stack                                                   # here, never at module import: the stock caller imports this module before its environment proof, which lists pxdesign_opt.stack among the modules that must be absent
     runner = I.InferenceRunner(configs)                                   # loads the model and the checkpoint (the hook fires here under a kit mode)
     if on_runner is not None:
         on_runner(runner)
'''
_LOCAL_PATCH = {
    "id": "pxdesign-stock-isolation-unused-import-v1",
    "base_kit_commit": _KIT_SHA,
    "file": "pxdesign/opt/pxdesign_opt/infer_loop.py",
    "before_sha256": "6b7f5b5cf251c38e452b2ab063f544f43d79b48684414d726b0f46e2743f638d",
    "after_sha256": "4e489b454adcc33cd67f497253107042a572a47c28e120563f7c26a6512fbe32",
    "diff_sha256": "39c8e5f7bd83ceb735c0771bb041fd9ea68e56766185acab3547a75b17885a68",
}
META.update({"source_kit_commit": _KIT_SHA, "local_patches": [_LOCAL_PATCH]})

_APPLY_STOCK_ISOLATION_PATCH = f'''python - <<'PX_STOCK_ISOLATION_REPAIR'
import ast, hashlib, json
from pathlib import Path
patch = {_STOCK_ISOLATION_PATCH!r}
audit = {_LOCAL_PATCH!r}
p = Path("/kit") / audit["file"]
source = p.read_bytes()
assert hashlib.sha256(source).hexdigest() == audit["before_sha256"], "PX driver source drift before local repair"
assert hashlib.sha256(patch.encode()).hexdigest() == audit["diff_sha256"], "PX local repair diff drift"
run = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "run")
assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == "stack" for n in ast.walk(run)), "PX driver now uses stack"
imports = [n for n in ast.walk(run) if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module is None and [a.name for a in n.names] == ["stack"]]
assert len(imports) == 1, "PX unused import is no longer unique"
lines = source.splitlines(keepends=True)
assert imports[0].lineno == imports[0].end_lineno, "PX import spans multiple lines"
repaired = b"".join(line for i, line in enumerate(lines, 1) if i != imports[0].lineno)
assert hashlib.sha256(repaired).hexdigest() == audit["after_sha256"], "PX driver source drift after local repair"
p.write_bytes(repaired)
Path("/kit/pxdesign/local-patches.json").write_text(json.dumps({{"source_kit_commit": audit["base_kit_commit"], "local_patches": [audit]}}, indent=2, sort_keys=True) + "\\n")
print("[pxdesign-local-patch] " + json.dumps(audit, sort_keys=True))
PX_STOCK_ISOLATION_REPAIR'''


def build(*, secrets: dict[str, str] | None = None) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the pinned upstream image, including stock's CUDA LayerNorm.

The compiler builds the same architectures and code as upstream. Build hosts
include GitHub, PyPI and download.pytorch.org. PXDesignBench keeps its .git
metadata because the upstream pin checker uses it. The unused upstream web UI
fonts are removed in the installation layer, as in the upstream Dockerfile.
"""
    del secrets
    runtime_env = {
        "PXDESIGN_CKPT_DIR": "/weights/pxdesign/checkpoint",
        "PROTENIX_DATA_ROOT_DIR": "/weights/pxdesign/ccd_cache",
        "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
        "TRITON_CACHE_DIR": "/weights/triton",
    }
    image = (
        # Bootstrap before Modal's registry-layer Python/pip validation.
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-devel-ubuntu22.04",
            setup_dockerfile_commands=[
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*",
                f"RUN curl -fsSL -o /tmp/python.tar.gz '{_PYTHON_ARCHIVE}' && "
                f"echo '{_PYTHON_DIGEST}  /tmp/python.tar.gz' | sha256sum -c - && "
                "tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && "
                "rm -rf /tmp/python /tmp/python.tar.gz && "
                "ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V",
            ],
        )
        .env({
            "PATH": "/usr/local/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONHASHSEED": "0", "CFLAGS": "-g0", **runtime_env,
        })
        .run_commands(
            "git init /kit_src && cd /kit_src && "
            f"git remote add origin {_KIT_REPO} && git config core.sparseCheckout true && "
            "printf '/pxdesign/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' > .git/info/sparse-checkout && "
            f"git fetch --depth 1 --filter=blob:none origin {_KIT_SHA} && git checkout FETCH_HEAD && "
            f"test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && "
            "mkdir -p /kit/common && mv pxdesign /kit/pxdesign && "
            "mv common/opt_core /kit/common/opt_core && "
            "cp LICENSE NOTICE /kit/ && cd / && rm -rf /kit_src",
        )
        .run_commands(
            f"cd {_KIT_ROOT} && grep -v '^#' environment/requirements.lock > /tmp/stack.txt && "
            "python -m pip install --no-cache-dir --no-deps --src /opt -r /tmp/stack.txt && "
            "rm /tmp/stack.txt && rm -rf /opt/pxdesign/.git && "
            "rm /opt/pxdesign/pxdesign/pxd_server/Helvetica-Regular.ttf "
            "/opt/pxdesign/pxdesign/pxd_server/TimesNewRoman.ttf",
            "J=$(nproc --all); MAX_JOBS=$J CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_THREADS=4 "
            "python -c \"import os, protenix.model.layer_norm.layer_norm as m; "
            "p = os.path.join(os.path.dirname(m.__file__), 'fastfold_layer_norm_cuda.so'); "
            "assert os.path.isfile(p), p; print('compiled', p)\"",
            f"cd {_KIT_ROOT} && bash run.sh install",
            _APPLY_STOCK_ISOLATION_PATCH,
        )
    )
    return image, {
        _WEIGHTS_MOUNT: modal.Volume.from_name(_WEIGHTS_VOLUME_NAME, create_if_missing=True),
    }, runtime_env


HYDRATE = (
    "bash", "-c",
    f"set -e; mkdir -p /weights/pxdesign {_JIT_CACHE_ROOT} /weights/triton; cd {_KIT_ROOT}; "
    "bash run.sh install --weights /weights/pxdesign",
)

KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ("off", "exact", "fast", "big")


def check_command(*, card: str = "h100", mode: str = "exact") -> tuple[str, ...]:
    """GPU-attached dry run after hydration; no design is generated."""
    if card not in KIT_CARDS:
        raise ValueError(f"unknown card {card!r}: {KIT_CARDS}")
    if mode not in KIT_MODES:
        raise ValueError(f"unknown mode {mode!r}: {KIT_MODES}")
    return ("bash", "-c", f"cd {_KIT_ROOT} && bash run.sh check --config {card} --mode {mode}")


CHECK = check_command()
