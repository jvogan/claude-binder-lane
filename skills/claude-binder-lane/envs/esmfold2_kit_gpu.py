"""Pinned H100 Modal environment for Anthropic's ESMFold2 optimization kit.

Lifecycle: prepare_wheels() on three sequential CPU Sandboxes, build() for the
runtime image, HYDRATE on a mounted weights Volume, CHECK on H100, then pred.
Importing this module and calling build() declare resources; neither starts a
cloud job. prepare_wheels() does start billed Modal Sandboxes.
"""

from __future__ import annotations

import modal


META = {
    "packages": ["torch", "triton", "esm", "transformers", "esmfold2_opt",
                 "opt_core", "flash_attn", "transformer_engine", "xformers"],
    "gpu_default": "H100",
    "supersedes": [],
    "egress_domains": ["huggingface.co", "*.hf.co"],
}

_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_KIT_ROOT = "/kit"
_STACK = "img_ef2_fa"
_WHEELS_MOUNT = "/wheels"
_WHEELS_VOLUME_NAME = "claude-science-esmfold2-kit-wheels-h100"
_WHEEL_DIR = f"{_WHEELS_MOUNT}/{_KIT_SHA}/{_STACK}"
_WEIGHTS_MOUNT = "/weights"
_WEIGHTS_VOLUME_NAME = "claude-science-esmfold2-kit-weights"
_JIT_CACHE_ROOT = "/weights/jit"

# Planning defaults, not observations. The caller can select smaller or larger
# resources after checking its own account limits, rates, and spend ceiling.
STAGE_RESOURCES = {
    "xformers": {"cpu": 8.0, "memory": 65536, "jobs": 8},
    "transformer_engine": {"cpu": 8.0, "memory": 65536, "jobs": 8},
    "flash_attn": {"cpu": 16.0, "memory": 131072, "jobs": 8},
}
_STAGES = tuple(STAGE_RESOURCES)


def _base_image() -> "modal.Image":
    """Lock and source shared by the build Sandboxes and finished image."""
    return (
        modal.Image.from_registry("python:3.12.10-slim-bookworm")
        .apt_install("gcc", "gfortran", "build-essential", "git", "curl",
                     "ca-certificates")
        .run_commands(
            f"git init {_KIT_ROOT}_src && cd {_KIT_ROOT}_src && "
            f"git remote add origin {_KIT_REPO} && "
            "git config core.sparseCheckout true && "
            "printf '/esmfold2/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' > .git/info/sparse-checkout && "
            f"git fetch --depth 1 origin {_KIT_SHA} && "
            f"git checkout FETCH_HEAD && test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && "
            f"mkdir -p {_KIT_ROOT}/common && "
            f"mv {_KIT_ROOT}_src/esmfold2 {_KIT_ROOT}/esmfold2 && "
            f"mv {_KIT_ROOT}_src/common/opt_core {_KIT_ROOT}/common/opt_core && "
            f"mv {_KIT_ROOT}_src/LICENSE {_KIT_ROOT}_src/NOTICE {_KIT_ROOT}/ && "
            f"rm -rf {_KIT_ROOT}_src",
        )
        .run_commands(
            f"cd {_KIT_ROOT}/esmfold2 && "
            "grep -E '^pip==' environment/requirements.lock | "
            "xargs python -m pip install --no-cache-dir --no-deps && "
            "grep -v -E '^(#|(flash_attn|transformer_engine|xformers)==)' "
            "environment/requirements.lock > /tmp/stack.txt && "
            "python -m pip install --no-cache-dir --no-deps -r /tmp/stack.txt && "
            "rm /tmp/stack.txt",
        )
        .env({
            "PYTHONHASHSEED": "0", "CFLAGS": "-g0", "NVTE_FRAMEWORK": "pytorch",
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": _WEIGHTS_MOUNT,
            "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
        })
    )


def _builder_image() -> "modal.Image":
    """Compile-only image; CUDA toolkit never enters the runtime image."""
    return _base_image().run_commands(
        "set -eu; curl -fsSL -o /tmp/cuda-keyring.deb "
        "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; "
        "echo 'e7f219eab6fe4819cdb5c15b98233dc3420302d9c00883219cd3d896857cf48d  /tmp/cuda-keyring.deb' | sha256sum -c -; "
        "dpkg -i /tmp/cuda-keyring.deb; apt-get update; "
        "apt-get install -y --no-install-recommends "
        "cuda-compiler-13-0 cuda-libraries-dev-13-0 cuda-cudart-dev-13-0 "
        "cuda-nvtx-13-0 cuda-profiler-api-13-0 cuda-nvml-dev-13-0; "
        "python -m pip install --no-cache-dir cmake==4.4.3 ninja "
        "'pybind11[global]==3.1.0' nvidia-cudnn-frontend==1.28.0",
    )


def _install_wheels(wheel_dir: str = _WHEEL_DIR) -> None:
    """Image build step: verify the wheel Volume before installing it."""
    import hashlib
    import pathlib
    import subprocess

    wheel_dir = pathlib.Path(wheel_dir)
    names = ("xformers", "transformer_engine", "flash_attn")
    wheels = []
    for name in names:
        digest_file = wheel_dir / f"{name}.sha256"
        lines = digest_file.read_text().splitlines()
        if len(lines) != 1:
            raise RuntimeError(f"missing or ambiguous digest: {digest_file}")
        digest, filename = lines[0].split(maxsplit=1)
        if filename.startswith("*"):
            filename = filename[1:]
        if pathlib.Path(filename).name != filename or not filename.startswith(name + "-") or not filename.endswith(".whl"):
            raise RuntimeError(f"unexpected wheel name in {digest_file}: {filename}")
        path = wheel_dir / filename
        hash_state = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hash_state.update(block)
        actual = hash_state.hexdigest()
        if actual != digest:
            raise RuntimeError(f"wheel digest mismatch: {path}")
        wheels.append(str(path))
    subprocess.run(["python", "-m", "pip", "install", "--no-cache-dir", "--no-index",
                    "--no-deps", *wheels], check=True)
    subprocess.run(["python", "-m", "pip", "check"], check=True)
    subprocess.run(["bash", "run.sh", "install"], cwd="/kit/esmfold2", check=True)


def build(*, secrets: dict[str, str] | None = None) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the runtime image. Run prepare_wheels before building it."""
    del secrets
    wheels = modal.Volume.from_name(_WHEELS_VOLUME_NAME, create_if_missing=False)
    image = _base_image().run_function(
        _install_wheels, volumes={_WHEELS_MOUNT: wheels}, cpu=2.0,
        memory=8192, timeout=3600, force_build=True,
    )
    weights = modal.Volume.from_name(_WEIGHTS_VOLUME_NAME, create_if_missing=True)
    return image, {_WEIGHTS_MOUNT: weights}, {
        "HF_HOME": _WEIGHTS_MOUNT, "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
    }


def wheel_stage_command(stage: str, *, jobs: int = 8) -> str:
    """Build one pinned wheel into a temporary directory, then publish its digest."""
    if stage not in _STAGES:
        raise ValueError(f"unknown wheel stage: {stage}")
    if not isinstance(jobs, int) or not 1 <= jobs <= 64:
        raise ValueError("jobs must be an integer from 1 through 64")
    head = (
        "set -euo pipefail; "
        "export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH "
        "TORCH_CUDA_ARCH_LIST=9.0 NVTE_FRAMEWORK=pytorch "
        "XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 NVCC_THREADS=2 "
        f"MAX_JOBS={jobs} CMAKE_BUILD_PARALLEL_LEVEL={jobs} MAKEFLAGS=-j{jobs}; "
        "test \"$(python -c 'import torch; print(torch.__version__)')\" = '2.13.0+cu130'; "
        "nvcc --version | grep -q 'release 13\\.'; "
        "mkdir -p /tmp/src /tmp/wheel; "
    )
    if stage == "xformers":
        body = (
            "git clone --quiet --depth 1 --branch v0.0.35 --recurse-submodules --shallow-submodules "
            "https://github.com/facebookresearch/xformers.git /tmp/src/xformers; "
            "test \"$(git -C /tmp/src/xformers rev-parse HEAD)\" = 03b91d7d9ff295ae68a320e2e733dd6c2ef8f342; "
            "cd /tmp/src/xformers; BUILD_VERSION=0.0.35+03b91d7.d20260904 "
            "FORCE_CUDA=1 XFORMERS_BUILD_TYPE=Release python -m pip wheel "
            "--no-cache-dir --no-build-isolation --no-deps -w /tmp/wheel .; "
        )
    elif stage == "transformer_engine":
        body = (
            "git clone --quiet --depth 1 --branch v2.15 --recurse-submodules --shallow-submodules "
            "https://github.com/NVIDIA/TransformerEngine.git /tmp/src/TransformerEngine; "
            "test \"$(git -C /tmp/src/TransformerEngine rev-parse HEAD)\" = 42b840051647eef89761a16dfdff87e82bb253ab; "
            "NCCL_HOME=$(python -c 'import nvidia.nccl; print(list(nvidia.nccl.__path__)[0])'); "
            "CUDNN_HOME=$(python -c 'import nvidia.cudnn; print(list(nvidia.cudnn.__path__)[0])'); "
            "test -f \"$NCCL_HOME/include/nccl.h\"; test -f \"$CUDNN_HOME/include/cudnn.h\"; "
            "export CPATH=$NCCL_HOME/include:$CUDNN_HOME/include LIBRARY_PATH=$NCCL_HOME/lib "
            f"NVTE_CUDA_ARCHS=90 NVTE_WITH_NCCL_EP=0 NVTE_BUILD_MAX_JOBS={jobs} CUDNN_PATH=$CUDNN_HOME; "
            "cd /tmp/src/TransformerEngine; python -m pip wheel --no-cache-dir "
            "--no-build-isolation --no-deps -w /tmp/wheel .; "
        )
    else:
        body = (
            "python -m pip download --no-cache-dir --no-binary :all: --no-deps "
            "--no-build-isolation -d /tmp/src flash-attn==2.8.3.post1; "
            "echo '55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348  /tmp/src/flash_attn-2.8.3.post1.tar.gz' | sha256sum -c -; "
            "MEM_GB=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); "
            f"JFA=$(( MEM_GB / 9 )); [ \"$JFA\" -ge 1 ] || JFA=1; [ \"$JFA\" -le {jobs} ] || JFA={jobs}; "
            "FLASH_ATTN_CUDA_ARCHS=90 FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=$JFA "
            "python -m pip wheel --no-cache-dir --no-build-isolation --no-deps "
            "-w /tmp/wheel /tmp/src/flash_attn-2.8.3.post1.tar.gz; "
        )
    tail = (
        "shopt -s nullglob; found=(/tmp/wheel/" + stage + "-*.whl); "
        "test ${#found[@]} -eq 1; "
        f"mkdir -p {_WHEEL_DIR}; cp \"${{found[0]}}\" {_WHEEL_DIR}/; "
        f"cd {_WHEEL_DIR}; sha256sum \"$(basename \"${{found[0]}}\")\" > {stage}.sha256; "
        f"sha256sum -c {stage}.sha256"
    )
    return head + body + tail


def prepare_wheels(*, app_name: str = "claude-science-esmfold2-kit",
                   resources: dict[str, dict[str, int | float]] | None = None) -> None:
    """Run three sequential, billed CPU Sandboxes with a persistent wheel Volume."""
    selected = resources or STAGE_RESOURCES
    for stage in _STAGES:
        config = selected[stage]
        if config["cpu"] <= 0 or config["memory"] <= 0:
            raise ValueError(f"invalid resources for {stage}")
        wheel_stage_command(stage, jobs=int(config["jobs"]))
    app = modal.App.lookup(app_name, create_if_missing=True)
    volume = modal.Volume.from_name(_WHEELS_VOLUME_NAME, create_if_missing=True)
    image = _builder_image()
    for stage in _STAGES:
        config = selected[stage]
        sandbox = modal.Sandbox.create(
            "sleep", "86400",
            app=app, image=image, volumes={_WHEELS_MOUNT: volume},
            cpu=float(config["cpu"]), memory=int(config["memory"]),
            timeout=24 * 60 * 60,
        )
        try:
            process = sandbox.exec("bash", "-lc", wheel_stage_command(stage, jobs=int(config["jobs"])),
                                   timeout=23 * 60 * 60, pty=True)
            for line in process.stdout:
                print(line, end="")
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(f"{stage} wheel build exited {process.returncode}")
        finally:
            sandbox.terminate(wait=True)
    # Modal commits each Volume on Sandbox shutdown. Subsequent image builds
    # mount the committed snapshot and _install_wheels rehashes every file.


# Upstream run.sh install --weights invokes esmfold2_opt.weights, which checks
# all 16 files against stock/PINS.json with fresh SHA-256 digests and exits 1
# on any missing or mismatched file. Each retry repeats the full check.
HYDRATE = (
    "bash", "-lc",
    f"set -e; mkdir -p {_WEIGHTS_MOUNT} {_JIT_CACHE_ROOT}; cd {_KIT_ROOT}/esmfold2; "
    f"for attempt in 1 2 3; do if bash run.sh install --weights {_WEIGHTS_MOUNT}; then exit 0; fi; "
    "echo \"weight hydration attempt $attempt failed\" >&2; sleep 10; done; exit 1",
)

KIT_CARDS = ("h100", "h200")
KIT_VARIANTS = ("fast", "full_msa", "full_nomsa")
KIT_MODES = ("off", "exact", "fast", "big")


def check_command(*, card: str = "h100", variant: str = "fast", mode: str = "fast") -> tuple[str, ...]:
    if card not in KIT_CARDS:
        raise ValueError(f"card {card!r} is not built into this image: {KIT_CARDS}")
    if variant not in KIT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}: {KIT_VARIANTS}")
    if mode not in KIT_MODES:
        raise ValueError(f"unknown mode {mode!r}: {KIT_MODES}")
    return ("bash", "-lc", f"cd {_KIT_ROOT}/esmfold2 && bash run.sh check "
            f"--config {card} --variant {variant} --mode {mode}")


CHECK = check_command()
