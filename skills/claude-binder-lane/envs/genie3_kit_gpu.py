"""Pinned Modal generation stack for Anthropic's Genie3 optimization kit.

build() declares an image and a persistent weights/JIT Volume. It launches no
job. HYDRATE fetches and hash-checks the upstream weights; CHECK needs a GPU.
All kit modes remain available. The stock Genie3 recipe is a separate route.
Source: genie3/environment/Dockerfile in uplifting-biomolecular-modeling at
f4f62fa6592ae4938d49b1757bea0cfeff9f468e.
"""

from __future__ import annotations

import modal


META = {
    "packages": ["torch", "triton", "lightning", "numpy", "genie3", "genie3_opt", "opt_core"],
    "gpu_default": "H100",
    "supersedes": [],
    "egress_domains": ["huggingface.co", "*.hf.co"],
}

_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_GENIE3_SHA = "d77ae5ac04212ff1e8b29b585859a3244c614804"
_KIT_ROOT = "/kit/genie3"
_WEIGHTS_MOUNT = "/weights"
_WEIGHTS_VOLUME_NAME = "claude-science-genie3-kit-weights"
_JIT_CACHE_ROOT = "/weights/jit"
_PYTHON_ARCHIVE = "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.10.13+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
_PYTHON_DIGEST = "ba512bcca3ac6cb6d834f496cd0a66416f0a53ff20b05c4794fa82ece185b85a"


def build(*, secrets: dict[str, str] | None = None) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the upstream Dockerfile's exact stack without downloading weights.

Keep the CUDA base's cuDNN libraries: the tested torch loader reads them before
its wheel copies. The Python archive is copied with links dereferenced as in
the upstream Dockerfile. The sparse source fetch retains upstream licensing.
Build-time network includes GitHub, PyPI and their artifact download hosts.
"""
    del secrets
    runtime_env = {
        "GENIE3_ROOT": "/opt/genie3",
        "GENIE3_WEIGHTS": "/weights/genie3/pretrained/v1",
        "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
    }
    image = (
        # Modal validates Python/pip in the registry layer before executing
        # later run_commands layers. Install the exact interpreter there.
        modal.Image.from_registry(
            "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04",
            setup_dockerfile_commands=[
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl git build-essential && rm -rf /var/lib/apt/lists/*",
                f"RUN curl -fsSL -o /tmp/python.tar.gz '{_PYTHON_ARCHIVE}' && "
                f"echo '{_PYTHON_DIGEST}  /tmp/python.tar.gz' | sha256sum -c - && "
                "mkdir /tmp/py && tar -xzf /tmp/python.tar.gz -C /tmp/py && "
                "cp -RL /tmp/py/python/. /usr/local/ && rm -rf /tmp/py /tmp/python.tar.gz && "
                "ln -sf /usr/local/bin/python3 /usr/local/bin/python && "
                "/usr/local/bin/python3 -m venv /opt/venv_g3",
            ],
        )
        .env({
            "PATH": "/opt/venv_g3/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONHASHSEED": "0", "CFLAGS": "-g0", **runtime_env,
        })
        .run_commands(
            "git init /kit_src && cd /kit_src && "
            f"git remote add origin {_KIT_REPO} && git config core.sparseCheckout true && "
            "printf '/genie3/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' > .git/info/sparse-checkout && "
            f"git fetch --depth 1 --filter=blob:none origin {_KIT_SHA} && git checkout FETCH_HEAD && "
            f"test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && "
            "mkdir -p /kit/common && mv genie3 /kit/genie3 && "
            "mv common/opt_core /kit/common/opt_core && "
            "cp LICENSE NOTICE /kit/ && cd / && rm -rf /kit_src",
        )
        .run_commands(
            f"cd {_KIT_ROOT} && "
            "python -m pip install --no-cache-dir \"$(grep '^pip==' environment/requirements.lock)\" && "
            "grep -v -E '^(#|-e )' environment/requirements.lock > /tmp/stack.txt && "
            "python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt && "
            "python -m pip install --no-cache-dir --no-deps --no-build-isolation --src /opt "
            "$(grep '^-e ' environment/requirements.lock) && python -m pip check && "
            f"test \"$(git -C /opt/genie3 rev-parse HEAD)\" = {_GENIE3_SHA} && "
            "rm -rf /opt/genie3/.git && rm /tmp/stack.txt && bash run.sh install",
        )
    )
    return image, {
        _WEIGHTS_MOUNT: modal.Volume.from_name(_WEIGHTS_VOLUME_NAME, create_if_missing=True),
    }, runtime_env


HYDRATE = (
    "bash", "-c",
    f"set -e; mkdir -p /weights/genie3 {_JIT_CACHE_ROOT}; cd {_KIT_ROOT}; "
    "bash run.sh install --weights /weights/genie3",
)

KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ("off", "exact", "fast")


def check_command(*, card: str = "h100", mode: str = "exact") -> tuple[str, ...]:
    """GPU-attached dry run after hydration; no design is generated."""
    if card not in KIT_CARDS:
        raise ValueError(f"unknown card {card!r}: {KIT_CARDS}")
    if mode not in KIT_MODES:
        raise ValueError(f"unknown mode {mode!r}: {KIT_MODES}")
    return ("bash", "-c", f"cd {_KIT_ROOT} && bash run.sh check --config {card} --mode {mode}")


CHECK = check_command()
