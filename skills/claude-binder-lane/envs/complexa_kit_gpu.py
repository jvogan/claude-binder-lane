"""Pinned Modal recipe for Anthropic's Proteina-Complexa optimization kit.

Preserves the upstream CPython 3.12.10 / torch 2.7.0+cu126 stack, bundled
Proteina-Complexa 1.1.0 archive, ColabDesign, and shared core. HYDRATE fetches
and digest-checks the two ~7 GB total checkpoints. CHECK needs the hydrated
weights and a GPU; a design proves the requested levers actually engaged.

The default upstream generation and AF2 reward workflow remains available
after HYDRATE_AF2. Configure actual optional tools when their reward or
evaluation stages are selected. Caller overrides pass verbatim and no reduced
scientific settings are added. An explicitly chosen single-pass/no-reward
workflow qualifies generation only.
"""

from __future__ import annotations

import shlex

import modal


META = {
    "packages": [
        "torch", "triton", "lightning", "hydra", "numpy", "jax",
        "proteinfoundation", "colabdesign", "complexa_opt", "opt_core",
    ],
    "gpu_default": "H100",
    "supersedes": [],
    "egress_domains": [
        # The bundled upstream downloader uses NGC, whose public downloads
        # redirect to xfiles. PINS names a matching HF source, not this route.
        "api.ngc.nvidia.com", "xfiles.ngc.nvidia.com",
        # Optional AF2 parameters are fetched only through HYDRATE_AF2.
        "storage.googleapis.com",
        "huggingface.co", "*.hf.co",
    ],
}
_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
_COMPLEXA_SHA = "916eaaedce5b07c205efb6ef32370c01d366591e"
_COMPLEXA_VERSION = "1.1.0"
_WEIGHT_DIGESTS = {
    "complexa.ckpt": "589db1741f29838c7961386f6b873087238c72682e56189b89e0ae02610c19e9",
    "complexa_ae.ckpt": "35f8865efd269995eeaf1670e1c1085acfe2988c40abdeda8e09a0e15eb40816",
}
_WEIGHT_BYTES = {"complexa.ckpt": 2934289381, "complexa_ae.ckpt": 4100101779}
_NGC_SOURCE = "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/clara/proteina_complexa/1.0/files?redirect=true&path="
_KIT_ROOT = "/kit/complexa"
_UPSTREAM_ROOT = "/opt/pc"
_WEIGHTS_MOUNT = "/weights"
_WEIGHTS_DIR = f"{_WEIGHTS_MOUNT}/complexa"
_AF2_DIR = f"{_WEIGHTS_MOUNT}/af2"
_JIT_CACHE_ROOT = f"{_WEIGHTS_MOUNT}/jit"
_WEIGHTS_VOLUME_NAME = "claude-science-complexa-kit-weights"
KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ("off", "exact", "fast", "big")


def build(
    *, secrets: dict[str, str] | None = None,
    volume_name: str = _WEIGHTS_VOLUME_NAME,
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Declare the complete pinned stack and an account-local weight Volume.

    The Dockerfile's PIP_CONSTRAINT pins isolated source-build backends,
    including hatchling 1.31.0. Omitting it may build a different environment.
    libxext6/libx11-6 are explicit because Modal apt installation may omit
    Debian's recommended packages that the Open Babel import requires.
    Build hosts include GitHub, PyPI, data.pyg.org, and storage.googleapis.com.
    """
    del secrets
    environment = {
        "PYTHONHASHSEED": "0",
        "CFLAGS": "-g0",
        "LOCAL_CODE_PATH": _UPSTREAM_ROOT,
        "CKPT_PATH": _WEIGHTS_DIR,
        "AF2_DIR": _AF2_DIR,
        "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
    }
    image = (
        modal.Image.from_registry("python:3.12.10-slim-bookworm")
        .apt_install(
            "build-essential", "git", "wget", "curl", "ca-certificates", "gfortran",
            "libxrender1", "libxext6", "libx11-6",
        )
        .run_commands(
            "git init /kit_src && cd /kit_src && "
            f"git remote add origin {_KIT_REPO} && "
            "git config core.sparseCheckout true && "
            "printf '/complexa/\\n/common/opt_core/\\n/LICENSE\\n/NOTICE\\n' "
            "> .git/info/sparse-checkout && "
            f"git fetch --depth 1 --filter=blob:none origin {_KIT_SHA} && "
            "git checkout FETCH_HEAD && "
            f"test \"$(git rev-parse HEAD)\" = {_KIT_SHA} && "
            "mkdir -p /kit/common && "
            "mv /kit_src/complexa /kit/complexa && "
            "mv /kit_src/common/opt_core /kit/common/opt_core && "
            "mv /kit_src/LICENSE /kit_src/NOTICE /kit/ && "
            "cd / && rm -rf /kit_src",
            f"grep -v '^-e ' {_KIT_ROOT}/environment/requirements.lock > /tmp/stack.txt && "
            "{ grep -E '^[A-Za-z0-9._-]+==' /tmp/stack.txt; "
            "echo 'hatchling==1.31.0'; } > /tmp/build-pins.txt && "
            "grep -E '^pip==' /tmp/stack.txt | "
            "xargs python3.12 -m pip install --no-cache-dir && "
            "PIP_CONSTRAINT=/tmp/build-pins.txt python3.12 -m pip install "
            "--no-cache-dir --no-deps -r /tmp/stack.txt",
            f"mkdir -p /opt && tar -xzf {_KIT_ROOT}/stock/"
            "proteina-complexa-916eaaed.tar.gz -C /opt && "
            f"mv /opt/proteina-complexa-916eaaed {_UPSTREAM_ROOT} && "
            "PIP_CONSTRAINT=/tmp/build-pins.txt python3.12 -m pip install "
            f"--no-cache-dir --no-deps -e {_UPSTREAM_ROOT} "
            f"-e {_UPSTREAM_ROOT}/community_models/colabdesign && "
            "rm -f /tmp/stack.txt /tmp/build-pins.txt",
        )
        .env(environment)
        .run_commands(f"cd {_KIT_ROOT} && bash run.sh install")
    )
    return image, {
        _WEIGHTS_MOUNT: modal.Volume.from_name(volume_name, create_if_missing=True),
    }, environment


# The stock downloader's GnuTLS transfer can fail after NGC redirects and leave
# a zero-byte final file. Stage the very same public sources with curl/OpenSSL,
# reacquiring the redirect on each bounded attempt. Never disable TLS, resume
# unidentified bytes, or publish a partial checkpoint. The unchanged kit
# installer still performs its own pinned size/SHA checks afterward.
_STAGE_WEIGHTS = r"""
stage_weight() {
    name="$1"; digest="$2"; bytes="$3"; url="$4"
    destination="${CKPT_PATH}/${name}"
    if [ -f "$destination" ] && [ "$(wc -c < "$destination")" -eq "$bytes" ] &&
       printf '%s  %s\n' "$digest" "$destination" | sha256sum -c >/dev/null 2>&1; then
        printf '[complexa hydrate] %s already verified\n' "$name"
        return 0
    fi
    candidate=$(mktemp "${CKPT_PATH}/.${name}.XXXXXX.part")
    for attempt in 1 2 3; do
        printf '[complexa hydrate] %s attempt %s/3\n' "$name" "$attempt"
        if curl --fail --location --silent --show-error --proto '=https' --proto-redir '=https' \
                --connect-timeout 30 --max-time 1800 --output "$candidate" "$url" &&
           [ "$(wc -c < "$candidate")" -eq "$bytes" ] &&
           printf '%s  %s\n' "$digest" "$candidate" | sha256sum -c >/dev/null 2>&1; then
            mv -f "$candidate" "$destination"
            printf '[complexa hydrate] %s verified and published atomically\n' "$name"
            return 0
        fi
        : > "$candidate"
    done
    rm -f "$candidate"
    printf '[complexa hydrate] refused %s after 3 verified-transfer attempts\n' "$name" >&2
    return 1
}
"""
HYDRATE = (
    "bash", "-lc",
    f"set -e; mkdir -p {_WEIGHTS_DIR} {_JIT_CACHE_ROOT}; "
    + _STAGE_WEIGHTS
    + "\n".join(
        "stage_weight " + shlex.join((name, _WEIGHT_DIGESTS[name], str(size), _NGC_SOURCE + name))
        for name, size in _WEIGHT_BYTES.items()
    )
    + f"\ncd {_KIT_ROOT}; bash run.sh install --weights {_WEIGHTS_DIR}",
)


# Optional full-workflow hydration. The upstream script locates downloads
# from its own directory, so a byte copy under a scratch project plus a
# symlink makes it write to the Volume without modifying the pinned checkout.
# Its own tar integrity and model-file checks remain in force. This is a
# separate billed operation; ordinary HYDRATE downloads Complexa only.
HYDRATE_AF2 = (
    "bash", "-lc",
    f"set -e; mkdir -p {_AF2_DIR}; "
    "scratch=$(mktemp -d); trap 'rm -rf \"$scratch\"' EXIT; "
    "mkdir -p \"$scratch/env\" \"$scratch/community_models/ckpts\"; "
    f"cp {_UPSTREAM_ROOT}/env/download_startup.sh \"$scratch/env/\"; "
    f"ln -s {_AF2_DIR} \"$scratch/community_models/ckpts/AF2\"; "
    "bash \"$scratch/env/download_startup.sh\" --af2; "
    "for model in 1 2 3 4 5; do "
    "for suffix in '' _ptm _multimer_v3; do "
    f"test -s \"{_AF2_DIR}/params_model_${{model}}${{suffix}}.npz\"; "
    "done; done",
)


def _selection(*, card: str, mode: str) -> None:
    if card not in KIT_CARDS:
        raise ValueError(f"unsupported kit card {card!r}; choose {KIT_CARDS}")
    if mode not in KIT_MODES:
        raise ValueError(f"unsupported kit mode {mode!r}; choose {KIT_MODES}")


def check_command(*, card: str = "h100", mode: str = "exact") -> tuple[str, ...]:
    """GPU-attached pin/weight/hook dry run after HYDRATE; launches no design."""
    _selection(card=card, mode=mode)
    return (
        "bash", "-lc",
        f"cd {_KIT_ROOT} && bash run.sh check --config {card} --mode {mode}",
    )


def design_command(
    *, out: str, mode: str, card: str = "h100", input_path: str | None = None,
    overrides: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Preserve every caller Hydra token and full upstream workflow by default.

    Single-pass/no-reward, seed, batch size, and sample count are never injected.
    Without input_path upstream selects its target from its config or overrides.
    """
    _selection(card=card, mode=mode)
    argv = ["bash", "run.sh", "design", "--config", card, "--mode", mode, "--out", out]
    if input_path is not None:
        argv.extend(("--input", input_path))
    if overrides:
        argv.extend(("--", *overrides))
    return "bash", "-lc", f"cd {_KIT_ROOT} && {shlex.join(argv)}"


CHECK = check_command()
