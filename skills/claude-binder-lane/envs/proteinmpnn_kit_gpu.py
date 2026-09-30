"""Pinned Modal recipe for Anthropic's full-backbone ProteinMPNN kit.

build() declares an image and Volume. HYDRATE fetches the pinned upstream
clone, including both weight sets, and verifies its digests. CHECK needs a GPU
and the hydrated clone. The first exact design compiles a Triton draw kernel
and captures CUDA graphs; its compile cache survives on the weights Volume.

This preserves the upstream Dockerfile's CUDA 12.4.1 base, hash-checked CPython
3.11.5 build, full requirements-mpnn.lock, shared core, and kit. No model code
or weights are baked into the image. The recipe adds a persistent cache and
fetches the release tree instead of requiring a local Docker build context.

exact supports vanilla and soluble full-backbone design. It refuses C-alpha
only, scoring-only, probability-only, tied-position, and noisy-backbone passes.
Those upstream capabilities remain accessible with an explicitly selected off
mode or a separately selected native route. No mode is silently substituted.
"""

from __future__ import annotations

import shlex

import modal


META = {
    "packages": ["torch", "triton", "numpy", "Bio", "proteinmpnn_opt", "opt_core"],
    "gpu_default": "H100",
    "supersedes": [],
    # HYDRATE fetches upstream's repository. Prediction needs no network.
    "egress_domains": ["github.com"],
}

_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_MPNN_SHA = "8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
_PYTHON_URL = (
    "https://github.com/indygreg/python-build-standalone/releases/download/20230826/"
    "cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz"
)
_PYTHON_SHA256 = "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1"
_WEIGHT_DIGESTS = {
    "vanilla_model_weights/v_48_020.pt": "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd",
    "soluble_model_weights/v_48_020.pt": "7af52d090172c230c7f0e9d21e02203f6b3a38b16db58d3c7a3960e0a9a6e31a",
}
_KIT_ROOT = "/kit/proteinmpnn"
_WEIGHTS_MOUNT = "/weights"
_MPNN_DIR = f"{_WEIGHTS_MOUNT}/ProteinMPNN"
_JIT_CACHE_ROOT = f"{_WEIGHTS_MOUNT}/jit"
_WEIGHTS_VOLUME_NAME = "claude-science-proteinmpnn-kit-weights"
KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ("off", "exact")
KIT_VARIANTS = ("vanilla", "soluble")


def build(
    *, secrets: dict[str, str] | None = None,
    volume_name: str = _WEIGHTS_VOLUME_NAME,
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the pinned image and account-local persistent source/weight cache.

    Image assembly and HYDRATE are billed when a provider builds/runs them.
    Build-only hosts are GitHub, its release assets, PyPI, and download.pytorch.org.
    """
    del secrets
    environment = {
        "PYTHONHASHSEED": "0",
        "MPNN_DIR": _MPNN_DIR,
        "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
    }
    image = (
        modal.Image.from_registry("nvidia/cuda:12.4.1-runtime-ubuntu22.04")
        .apt_install("ca-certificates", "curl", "git", "build-essential")
        .run_commands(
            f"curl -fsSL -o /tmp/python.tar.gz {shlex.quote(_PYTHON_URL)} && "
            f"echo '{_PYTHON_SHA256}  /tmp/python.tar.gz' | sha256sum -c - && "
            "tar -xzf /tmp/python.tar.gz -C /tmp && "
            "cp -RL /tmp/python/. /usr/local/ && "
            "rm -rf /tmp/python /tmp/python.tar.gz && "
            "ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V",
            "git init /kit_src && cd /kit_src && "
            f"git remote add origin {_KIT_REPO} && "
            "git config core.sparseCheckout true && "
            "printf '/proteinmpnn/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' "
            "> .git/info/sparse-checkout && "
            f"git fetch --depth 1 --filter=blob:none origin {_KIT_SHA} && "
            "git checkout FETCH_HEAD && "
            f"test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && "
            "mkdir -p /kit/common && "
            "mv /kit_src/proteinmpnn /kit/proteinmpnn && "
            "mv /kit_src/common/opt_core /kit/common/opt_core && "
            "mv /kit_src/LICENSE /kit_src/NOTICE /kit/ && "
            "cd / && rm -rf /kit_src",
            f"python -m pip install --no-cache-dir --no-deps "
            f"-r {_KIT_ROOT}/environment/requirements-mpnn.lock",
            "python -m pip install --no-cache-dir "
            "-e /kit/common/opt_core -e /kit/proteinmpnn/opt && "
            "python -c 'from proteinmpnn_opt import core_gate; core_gate()'",
        )
        .env(environment)
    )
    return image, {
        _WEIGHTS_MOUNT: modal.Volume.from_name(volume_name, create_if_missing=True),
    }, environment


HYDRATE = (
    "bash", "-lc",
    f"set -e; mkdir -p {_WEIGHTS_MOUNT} {_JIT_CACHE_ROOT}; "
    f"cd {_KIT_ROOT}; bash run.sh install --weights {_MPNN_DIR}",
)


def _selection(*, card: str, mode: str, variant: str) -> None:
    if card not in KIT_CARDS:
        raise ValueError(f"unsupported kit card {card!r}; choose {KIT_CARDS}")
    if mode not in KIT_MODES:
        raise ValueError(f"unsupported kit mode {mode!r}; choose {KIT_MODES}")
    if variant not in KIT_VARIANTS:
        raise ValueError(f"unsupported kit variant {variant!r}; choose {KIT_VARIANTS}")


def check_command(
    *, card: str = "h100", mode: str = "exact", variant: str = "vanilla",
) -> tuple[str, ...]:
    """GPU-attached dry run after HYDRATE; probes actual acceleration at design.

    Upstream check accepts exact only. The off route is qualified by a design.
    """
    _selection(card=card, mode=mode, variant=variant)
    if mode != "exact":
        raise ValueError("upstream check supports exact only; use design for off")
    return (
        "bash", "-lc",
        f"cd {_KIT_ROOT} && bash run.sh check --config {card} "
        f"--mode {mode} --variant {variant}",
    )


def design_command(
    *, input_path: str, out: str, mode: str, card: str = "h100",
    variant: str = "vanilla", options: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Pass the scientist's upstream options unchanged, including unsupported passes.

    The upstream kit issues its named refusal when exact cannot serve a pass.
    Request counts, batch sizes, temperatures, constraints, and scientific
    parameters are the caller's choices. No allow-partial or opt-out is added.
    """
    _selection(card=card, mode=mode, variant=variant)
    argv = (
        "bash", "run.sh", "design", "--config", card, "--mode", mode,
        "--variant", variant, "--input", input_path, "--out", out, *options,
    )
    return "bash", "-lc", f"cd {_KIT_ROOT} && {shlex.join(argv)}"


CHECK = check_command()
