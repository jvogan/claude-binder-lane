"""Pinned, account-independent Modal recipe for Anthropic's BoltzGen kit.

The image preserves the upstream Dockerfile's base, standalone CPython build,
full requirements.lock, bundled stock distribution, and kit installation. The
kit release is pinned by full Git object name. Weights and the persistent JIT
cache live on the caller's Modal account, outside the image. Importing this
module only declares commands; building/hydrating requires authorized compute.

Qualification status: the recipe has offline contract tests. GPU execution
and acceleration claims require a real run receipt, parsed artifacts, and the
kit's requested-mode activation evidence. CHECK needs an attached GPU after
HYDRATE; a DRY-RUN line proves readiness, not applied acceleration.

See ../references/boltzgen-kit-modal.md for commands, capability boundaries, and
workflow handoffs. Upstream pins and installation commands are from:
https://github.com/anthropics/uplifting-biomolecular-modeling/blob/
f4f62fa6592ae4938d49b1757bea0cfeff9f468e/boltzgen/environment/Dockerfile
"""

from __future__ import annotations

import shlex

import modal

_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_KIT_ROOT = "/kit/boltzgen"
_WEIGHTS_MOUNT = "/weights"
_WEIGHTS_ROOT = "/weights/boltzgen"
_JIT_CACHE_ROOT = "/weights/jit/boltzgen"
_WEIGHTS_VOLUME_NAME = "claude-science-boltzgen-kit-weights"
KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ('off', 'exact', 'fast', 'big')

META = {
    "packages": ['torch', 'triton', 'boltzgen', 'cuequivariance', 'cuequivariance_ops_torch', 'boltzgen_opt', 'opt_core'],
    "gpu_default": "H100",
    "supersedes": [],
    "egress_domains": ['huggingface.co', '*.hf.co'],
}

# These two command bodies are the pinned upstream Dockerfile RUN steps.
# The standalone Python archive and source distributions retain their digests.
_PYTHON_BOOTSTRAP = r"""curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20230826/cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" \
 && echo "fbed6f7694b2faae5d7c401a856219c945397f772eea5ca50c6eb825cbc9d1e1  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -a /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V"""
_STACK_INSTALL = r"""echo "build cores: $(nproc --all)" && grep -E '^(pip|setuptools|wheel)==' /tmp/requirements.lock | xargs python -m pip install --no-cache-dir --no-deps \
 && grep -v -E '^(#|boltzgen==)' /tmp/requirements.lock > /tmp/stack.txt && python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
 && python -m pip install --no-cache-dir --no-deps /tmp/stock/boltzgen-0.3.2-py3-none-any.whl \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/requirements.lock"""


def _fetch_kit_command() -> str:
    # Fetch, sparse checkout, move, and cleanup share one layer. Root-anchored
    # paths avoid pulling another model's nested common or kit directory.
    return (
        "git init /tmp/kit-src && cd /tmp/kit-src && "
        f"git remote add origin {_KIT_REPO} && git config core.sparseCheckout true && "
        "printf '/boltzgen/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' > .git/info/sparse-checkout && "
        f"git fetch --filter=blob:none --depth 1 origin {_KIT_SHA} && git checkout FETCH_HEAD && "
        f"test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && mkdir -p /kit/common && "
        "mv boltzgen /kit/boltzgen && mv common/opt_core /kit/common/opt_core && "
        "mv LICENSE NOTICE /kit/ && cd / && rm -rf /tmp/kit-src"
    )


def build(*, secrets: dict[str, str] | None = None,
          weights_volume_name: str = _WEIGHTS_VOLUME_NAME
          ) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the genuine pinned image and an account-local weights Volume.

    Image assembly and HYDRATE can incur charges. The caller must establish the
    provider, budget, execution timeout, and output ownership before launching.
    Build hosts (GitHub, PyPI, download.pytorch.org and NVIDIA's apt repository)
    are distinct from META's running-job and hydration egress hosts.
    """
    del secrets
    image = (
        # CUDA's base has no Python. Bootstrap the exact release in the first
        # registry layer, before Modal validates or extends the image.
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-base-ubuntu22.04",
            setup_dockerfile_commands=[
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl build-essential git && rm -rf /var/lib/apt/lists/*",
                f"RUN {_PYTHON_BOOTSTRAP}",
            ],
        )
        .run_commands(_fetch_kit_command())
        .run_commands(
            f"cp {_KIT_ROOT}/environment/requirements.lock /tmp/requirements.lock && "
            "mkdir -p /tmp/stock && cp /kit/boltzgen/stock/boltzgen-0.3.2-py3-none-any.whl /tmp/stock/ && " + _STACK_INSTALL,
        )
        .run_commands(f"cd {_KIT_ROOT} && bash run.sh install")
        .env({"PYTHONHASHSEED": "0", "CFLAGS": "-g0",
              "BOLTZGEN_CACHE": _WEIGHTS_ROOT, "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT})
    )
    weights = modal.Volume.from_name(weights_volume_name, create_if_missing=True)
    return image, {_WEIGHTS_MOUNT: weights}, {"BOLTZGEN_CACHE": _WEIGHTS_ROOT, "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT}


HYDRATE = (
    "bash", "-c",
    f"set -e; mkdir -p {_WEIGHTS_ROOT} {_JIT_CACHE_ROOT}; cd {_KIT_ROOT}; "
    f"bash run.sh install --weights {_WEIGHTS_ROOT}",
)


def check_command(*, card: str = "h100", mode: str = "exact") -> tuple[str, ...]:
    """GPU-attached dry run after HYDRATE; no designs or applied levers."""
    if card not in KIT_CARDS:
        raise ValueError(f"unknown configured card {card!r}: {KIT_CARDS}")
    if mode not in KIT_MODES:
        raise ValueError(f"unknown kit mode {mode!r}: {KIT_MODES}")
    return (
        "bash", "-c",
        f"cd {_KIT_ROOT} && bash run.sh check --config {card} --mode {mode}",
    )


CHECK = check_command()


def smoke_command(*, output_dir: str,
                  input_spec: str = "/kit/boltzgen/opt/forward/fast_inference/tests/specs/pdl1_ref.yaml",
                  card: str = "h100", mode: str = "exact", seed: int = 0,
                  full_pipeline: bool = False) -> tuple[str, ...]:
    """Two real PD-L1 designs; optionally run upstream's whole pipeline.

    Exact and fast need a diffusion batch of at least two. Reducing it to one
    refuses the requested acceleration. This smoke preserves model sampling
    and recycling settings. A design-only run proves generation; use
    full_pipeline=True to exercise inverse folding, folding, and analysis too.
    """
    check_command(card=card, mode=mode)
    if not output_dir.startswith("/") or not input_spec.startswith("/"):
        raise ValueError("input_spec and output_dir must be absolute worker paths")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(full_pipeline, bool):
        raise ValueError("full_pipeline must be a boolean")
    argv = ["bash", "run.sh", "design", "--config", card, "--mode", mode,
            input_spec, "--output", output_dir, "--num_designs", "2", "--seed",
            str(seed), "--diffusion_batch_size", "2"]
    if not full_pipeline:
        argv += ["--steps", "design"]
    return ("bash", "-c", f"cd {_KIT_ROOT} && {shlex.join(argv)}")
