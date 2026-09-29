"""Operator-configured Protenix v2 recipe plus ipSAE and DockQ scoring tools.

This recipe records the published third co-folding mode and the scoring stack that
would read its output after runtime qualification. The two ESMFold2 modes are not in here. They pin
requires-python ">=3.12,<3.13" and the compute skill already ships an `esmfold2_gpu`
environment carrying them, so they stay in their own image.

references/tool-bringup.md says how to build this and what PASS means.

Two design points come straight from bring-up experience, and both cost an afternoon
if you drop them.

The first Protenix v2 call compiles a CUDA kernel, four to six minutes on an H100.
That needs a `devel` base image for nvcc, and it needs the torch extensions cache on a
volume shared across containers, or every container pays the compile again.

Weights are hydrated into a volume at build time and read offline at run time. That is
the pattern the bundled `esmfold2_gpu` environment uses, and it keeps run-time egress
close to empty, which matters under an allowlist network policy.

The source and container family are pinned, but the official checkpoint object has
no upstream digest and returned HTTP 403 when fetched. `build()` refuses until a
committed dependency lock, immutable artifact manifest, and complete image recipe
exist. See `references/protenix-v2-bringup.md`.
"""

import modal

_PROTENIX_RELEASE = "v2.0.0"
_PROTENIX_PIN = "2475421477ab414b571149ad4a875c390ff8a35d"
_PROTENIX_BASE_IMAGE = (
    "vemlp-cn-beijing.cr.volces.com/preset-images/pytorch@"
    "sha256:bb558085abee1d77f0c981220a474d63f55d42143a83580116df67484932c604"
)
_PROTENIX_CUTLASS_PIN = "f7b19de32c5d1f3cedfc735c2849f12b537522ee"
_PROTENIX_WEIGHTS_URL = (
    "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt"
)
# Upstream publishes neither an object digest nor a signed artifact manifest.
# Capture the official bytes once, review the receipt, then commit the digest here.
_PROTENIX_WEIGHTS_SHA256 = None
_PROTENIX_REQUIREMENTS_LOCK_SHA256 = None
_PROTENIX_IMAGE_RECIPE_COMPLETE = False

# ipSAE is one Python script with numpy as its only stated dependency.
_IPSAE_URL = (
    "https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py"
)

# No commit pins ipsae.py. The line above fetches from the default branch,
# which is not reproducible. Record the commit the campaign scored against, because the
# d0res normalization is what `ipSAE_min` is computed from.

META = {
    "packages": ["protenix", "DockQ", "ipsae", "torch"],
    # Protenix v2 compiles its kernel on first call and the campaign scores at five
    # seeds per arm. 80 GB is the headroom the other proteomics envs on this install
    # settle on for co-folding work.
    "gpu_default": "A100-80GB",
    # Nothing here replaces a bundled environment.
    "supersedes": [],
    # Prediction consumes a precomputed target MSA and a prehydrated, hash-checked
    # Protenix cache. Hydration has a separate TOS allowlist; the paid GPU job has none.
    "egress_domains": [],
}

_ENV = {
    "PROTENIX_ROOT_DIR": "/datavol_binder/protenix",
    # The kernel compile lands here, on a volume, so it happens once per campaign
    # instead of once per container.
    "TORCH_EXTENSIONS_DIR": "/datavol_binder/torch_extensions",
    "IPSAE_SCRIPT": "/opt/ipsae/ipsae.py",
}


def build(
    *, secrets: dict[str, str] | None = None
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    secrets = secrets or {}
    if (
        not _PROTENIX_IMAGE_RECIPE_COMPLETE
        or _PROTENIX_REQUIREMENTS_LOCK_SHA256 is None
        or _PROTENIX_WEIGHTS_SHA256 is None
    ):
        raise RuntimeError(
            "binder_scoring_gpu: Protenix v2 remains fail-closed. Commit the reviewed "
            "requirements lock, official artifact manifest and SHA-256, then implement "
            "the pinned image recipe from references/protenix-v2-bringup.md."
        )
    img = (
        # devel, not runtime: the first Protenix v2 call needs nvcc.
        modal.Image.from_registry(_PROTENIX_BASE_IMAGE)
        .apt_install("git", "wget")
        .pip_install(
            "numpy",
            "DockQ",
            "huggingface_hub",
        )
        # ipSAE ships as a single script rather than a package.
        .run_commands(
            "mkdir -p /opt/ipsae",
            f"wget -q -O /opt/ipsae/ipsae.py {_IPSAE_URL}",
        )
        # This branch stays unreachable until the reviewed lock, artifact manifest,
        # and full pinned install recipe are committed together.
        .env(_ENV)
    )
    vols = {
        "/datavol_binder": modal.Volume.from_name(
            "claude-binder-lane-scoring-cache", create_if_missing=True
        ),
    }
    return img, vols, dict(_ENV)


# Runs once in a CPU sandbox when build_env(hydrate=True) is called. It refuses until
# the official artifact digest exists; an implicit Protenix auto-download is forbidden.
HYDRATE = (
    "python",
    "-c",
    "import sys\n"
    "url, digest = " + repr(_PROTENIX_WEIGHTS_URL) + ", " + repr(_PROTENIX_WEIGHTS_SHA256) + "\n"
    "if digest is None:\n"
    "  sys.exit('binder_scoring_gpu: official protenix-v2.pt has no reviewed SHA-256; "
    "capture the first-party artifact receipt before hydration: ' + url)\n"
    "sys.exit('binder_scoring_gpu: implement the complete hash-checked artifact "
    "manifest before enabling hydration')\n",
)
