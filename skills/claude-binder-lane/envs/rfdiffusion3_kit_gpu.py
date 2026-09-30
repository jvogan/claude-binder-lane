"""Pinned, account-independent Modal recipe for Anthropic's RFdiffusion3 kit.

The image preserves the upstream Dockerfile's base, standalone CPython build,
full requirements.lock, bundled stock distribution, and kit installation. The
kit release is pinned by full Git object name. Weights and the persistent JIT
cache live on the caller's Modal account, outside the image. Importing this
module only declares commands; building/hydrating requires authorized compute.

Qualification status: the recipe has offline contract tests. GPU execution
and acceleration claims require a real run receipt, parsed artifacts, and the
kit's requested-mode activation evidence. CHECK needs an attached GPU after
HYDRATE; a DRY-RUN line proves readiness, not applied acceleration.

See ../references/rfdiffusion3-kit-modal.md for commands, capability boundaries, and
workflow handoffs. Upstream pins and installation commands are from:
https://github.com/anthropics/uplifting-biomolecular-modeling/blob/
f4f62fa6592ae4938d49b1757bea0cfeff9f468e/rfdiffusion3/environment/Dockerfile
"""

from __future__ import annotations

import shlex

import modal

_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_KIT_ROOT = "/kit/rfdiffusion3"
_WEIGHTS_MOUNT = "/weights"
_WEIGHTS_ROOT = "/weights/rfd3"
_JIT_CACHE_ROOT = "/weights/jit/rfdiffusion3"
_WEIGHTS_VOLUME_NAME = "claude-science-rfdiffusion3-kit-weights"
KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ('off', 'exact', 'fast')

META = {
    "packages": ['torch', 'triton', 'rc-foundry', 'rfd3', 'apex', 'atomworks', 'rfdiffusion3_opt', 'opt_core'],
    "gpu_default": "H100",
    "supersedes": [],
    "egress_domains": ['files.ipd.uw.edu'],
}

# These two command bodies are the pinned upstream Dockerfile RUN steps.
# The standalone Python archive and source distributions retain their digests.
_PYTHON_BOOTSTRAP = r"""curl -fsSL -o /tmp/python.tar.gz "https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.12.1+20240107-x86_64_v3-unknown-linux-gnu-install_only.tar.gz" \
 && echo "a7bcc6c9f66dbd47ea99615f30f101c5d2dd0084ca333d2f7336e64050951338  /tmp/python.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/python.tar.gz -C /tmp && cp -RLp /tmp/python/. /usr/local/ && rm -rf /tmp/python /tmp/python.tar.gz \
 && ln -sf /usr/local/bin/python3 /usr/local/bin/python && python -V"""
_STACK_INSTALL = r"""grep -v -E '^(#|apex==|rc-foundry==)' /tmp/requirements.lock > /tmp/stack.txt \
 && printf 'hatchling==1.32.0\nhatch-vcs==0.4.0\nsetuptools-scm==10.2.1\nvcs-versioning==2.3.1\n' > /tmp/build-pins.txt \
 && eval "$(python -c "import json; p = json.load(open('/tmp/stock/PINS.json')); a = p['pinned_stack']['apex']; print('FOUNDRY_VERSION=%s APEX_WHEEL=%s APEX_SHA256=%s APEX_REPO=%s APEX_COMMIT=%s' % (p['upstream']['foundry']['version'], a['wheel'], a['wheel_sha256'], a['repo'], a['commit']))")" \
 && for venv in /opt/stock /opt/kit; do \
      /usr/local/bin/python -m venv "$venv" \
   && "$venv/bin/python" -m pip install --no-cache-dir --no-deps "$(grep -E '^pip==' /tmp/stack.txt)" \
   && "$venv/bin/python" -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt \
   && SETUPTOOLS_SCM_PRETEND_VERSION="$FOUNDRY_VERSION" PIP_CONSTRAINT=/tmp/build-pins.txt "$venv/bin/python" -m pip install --no-cache-dir --no-deps /tmp/stock/foundry-4010e3e.tar.gz \
   || exit 1; \
    done \
 && if [ "$WHEELS_FROM" = build ]; then \
      J=${BUILD_JOBS:-0}; [ "$J" -gt 0 ] || J=$(nproc --all); P=$(( J < 8 ? J : 8 )); MJ=$(( J / P )); [ "$MJ" -gt 0 ] || MJ=1; \
      echo "apex $APEX_COMMIT from source: $(nproc --all) cores, BUILD_JOBS=$J ($P extensions at a time, $MJ ninja jobs each, nvcc --threads 4), TORCH_CUDA_ARCH_LIST=8.0;9.0+PTX" \
   && apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0 \
        libcublas-dev-13-0 libcurand-dev-13-0 libcusparse-dev-13-0 libcusolver-dev-13-0 \
   && /usr/local/cuda-13.0/bin/nvcc --version | tail -1 \
   && curl -fsSL -o /tmp/apex.tar.gz "$APEX_REPO/archive/$APEX_COMMIT.tar.gz" && echo "f3cc6e36d357f5c47e39fe6f7ab7282ec7fc19af0b4a8425758ec2f2c6cc038e  /tmp/apex.tar.gz" | sha256sum -c - \
   && mkdir -p /tmp/src /tmp/stock/wheels && tar -xzf /tmp/apex.tar.gz -C /tmp/src \
   && /opt/stock/bin/python -m pip install --no-cache-dir --target /tmp/buildtools ninja==1.13.0 \
   && cd "/tmp/src/apex-$APEX_COMMIT" \
   && PYTHONPATH=/tmp/buildtools PATH=/tmp/buildtools/bin:/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 CC=gcc CXX=g++ \
      APEX_CPP_EXT=1 APEX_CUDA_EXT=1 APEX_PARALLEL_BUILD=$P MAX_JOBS=$MJ CMAKE_BUILD_PARALLEL_LEVEL=$J MAKEFLAGS=-j$J NVCC_APPEND_FLAGS="--threads 4" TORCH_CUDA_ARCH_LIST="8.0;9.0+PTX" \
      /opt/stock/bin/python -m pip wheel --no-deps --no-build-isolation --no-cache-dir -w /tmp/stock/wheels . \
   && cd / && echo "apex wheel compiled here (this build's own file, not the published one): $(sha256sum "/tmp/stock/wheels/$APEX_WHEEL")" \
   && apt-get purge -y --auto-remove cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0 \
        libcublas-dev-13-0 libcurand-dev-13-0 libcusparse-dev-13-0 libcusolver-dev-13-0 \
   && rm -rf /var/lib/apt/lists/* /tmp/src /tmp/buildtools /tmp/apex.tar.gz || exit 1; \
    elif [ "$WHEELS_FROM" = prebuilt ]; then echo "$APEX_SHA256  /tmp/stock/wheels/$APEX_WHEEL" | sha256sum -c - || exit 1; \
    else echo "WHEELS_FROM=$WHEELS_FROM: build or prebuilt" >&2; exit 2; fi \
 && for venv in /opt/stock /opt/kit; do \
      "$venv/bin/python" -m pip install --no-cache-dir --no-deps "/tmp/stock/wheels/$APEX_WHEEL" \
   && "$venv/bin/python" -c "import rfd3, apex, importlib; importlib.import_module('apex.normalization.fused_layer_norm'); import fused_layer_norm_cuda" \
   || exit 1; \
    done \
 && rm -rf /tmp/stock /tmp/stack.txt /tmp/build-pins.txt /tmp/requirements.lock /root/.cache"""


def _fetch_kit_command() -> str:
    # Fetch, sparse checkout, move, and cleanup share one layer. Root-anchored
    # paths avoid pulling another model's nested common or kit directory.
    return (
        "git init /tmp/kit-src && cd /tmp/kit-src && "
        f"git remote add origin {_KIT_REPO} && git config core.sparseCheckout true && "
        "printf '/rfdiffusion3/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' > .git/info/sparse-checkout && "
        f"git fetch --filter=blob:none --depth 1 origin {_KIT_SHA} && git checkout FETCH_HEAD && "
        f"test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && mkdir -p /kit/common && "
        "mv rfdiffusion3 /kit/rfdiffusion3 && mv common/opt_core /kit/common/opt_core && "
        "mv LICENSE NOTICE /kit/ && cd / && rm -rf /tmp/kit-src"
    )


def build(*, secrets: dict[str, str] | None = None,
          weights_volume_name: str = _WEIGHTS_VOLUME_NAME, build_jobs: int = 0
          ) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the genuine pinned image and an account-local weights Volume.

    Image assembly and HYDRATE can incur charges. The caller must establish the
    provider, budget, execution timeout, and output ownership before launching.
    Build hosts (GitHub, PyPI, download.pytorch.org and NVIDIA's apt repository)
    are distinct from META's running-job and hydration egress hosts.
    """
    del secrets
    if not isinstance(build_jobs, int) or isinstance(build_jobs, bool) or build_jobs < 0:
        raise ValueError("build_jobs must be zero (upstream default) or a positive integer")
    image = (
        # CUDA's base has no Python. Bootstrap the exact release in the first
        # registry layer, before Modal validates or extends the image.
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-runtime-ubuntu22.04",
            setup_dockerfile_commands=[
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl build-essential git && rm -rf /var/lib/apt/lists/*",
                f"RUN {_PYTHON_BOOTSTRAP}",
            ],
        )
        .run_commands(_fetch_kit_command())
        .env({"WHEELS_FROM": "build", "BUILD_JOBS": str(build_jobs)})
        .run_commands(
            f"cp {_KIT_ROOT}/environment/requirements.lock /tmp/requirements.lock && "
            "cp -a /kit/rfdiffusion3/stock /tmp/stock && " + _STACK_INSTALL,
        )
        .run_commands(
            f"cd {_KIT_ROOT} && PATH=/opt/stock/bin:$PATH bash run.sh install && "
            "PATH=/opt/kit/bin:$PATH bash run.sh install",
            f"cd {_KIT_ROOT} && /opt/kit/bin/python -m rfdiffusion3_opt.install apply && "
            "/opt/kit/bin/python -m rfdiffusion3_opt.install check-only",
        )
        .env({"PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1", "CFLAGS": "-g0",
              "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
              "PATH": "/opt/kit/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
              "RFDIFFUSION3_STOCK_PYTHON": "/opt/stock/bin/python",
              "RFD3_CKPT": f"{_WEIGHTS_ROOT}/rfd3_latest.ckpt",
              "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT})
    )
    weights = modal.Volume.from_name(weights_volume_name, create_if_missing=True)
    return image, {_WEIGHTS_MOUNT: weights}, {"RFD3_CKPT": f"{_WEIGHTS_ROOT}/rfd3_latest.ckpt",
        "RFDIFFUSION3_STOCK_PYTHON": "/opt/stock/bin/python",
        "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT}


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


def prepare_smoke_command(*, input_dir: str = "/work/smoke-inputs") -> tuple[str, ...]:
    """Stage the upstream PD-L1 binder spec and its public structure input.

    The source example also includes an insulin-receptor spec. Selecting just
    its PD-L1 entry makes this one real binder, preserving all scientific
    settings and resolving its relative structure path before dispatch.
    """
    if not input_dir.startswith("/"):
        raise ValueError("input_dir must be an absolute path on the worker")
    program = (
        "import json,pathlib,sys,tarfile; "
        "out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True); "
        "arc=tarfile.open('/kit/rfdiffusion3/stock/foundry-4010e3e.tar.gz'); "
        "prefix='foundry-4010e3e2/models/rfd3/docs/'; "
        "arc.extractall(out,members=[m for m in arc.getmembers() if m.name.startswith(prefix)],filter='data'); "
        "docs=out/'foundry-4010e3e2/models/rfd3/docs'; "
        "item=json.loads((docs/'examples/protein_binder_design.json').read_text())['pdl1']; "
        "item['input']=str((docs/'input_pdbs/5o45_cropped.pdb').resolve()); "
        "(out/'pdl1.json').write_text(json.dumps({'pdl1':item},indent=2)+'\\n')"
    )
    return ("python", "-c", program, input_dir)


def smoke_command(*, output_dir: str, input_spec: str = "/work/smoke-inputs/pdl1.json",
                  card: str = "h100", mode: str = "exact", seed: int = 101
                  ) -> tuple[str, ...]:
    """One real upstream PD-L1 binder with unchanged sampling parameters.

    A new output_dir is required: upstream skips existing RFdiffusion3 designs,
    and a run that performs no model work cannot prove kit activation.
    """
    check_command(card=card, mode=mode)
    if not output_dir.startswith("/") or not input_spec.startswith("/"):
        raise ValueError("input_spec and output_dir must be absolute worker paths")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    argv = ["bash", "run.sh", "design", "--config", card, "--mode", mode,
            f"inputs={input_spec}", f"out_dir={output_dir}", "n_batches=1",
            "diffusion_batch_size=1", f"seed={seed}"]
    return ("bash", "-c", f"cd {_KIT_ROOT} && {shlex.join(argv)}")
