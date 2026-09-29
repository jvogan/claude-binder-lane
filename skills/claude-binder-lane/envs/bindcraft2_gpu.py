"""BindCraft 2 on a Modal GPU sandbox, installed the way upstream installs it.

The operator's own installation is what the BindCraft2 licence permits, and a Modal
sandbox on the operator's own account is that. This package hosts nothing: it builds an
image in the operator's account, the operator's agent directs the run, and the design
outputs come back to them. `references/tool-catalogue.md` records the licence reading.

Three details decide whether the image works, and each one costs a rebuild if dropped.

`add_python` is 3.12 because upstream `pyproject.toml` sets requires-python to 3.12 or
newer. The compute skill's bundled `proteomics_jax_gpu` environment carries 3.11, so it
cannot host this package however close its JAX stack looks.

The install is editable. Upstream `pyproject.toml` declares only the ProteinMPNN
checkpoints as package data, and `bindcraft.cli` resolves its five preset vocabularies
from a `settings` directory beside the package, so a copy installed from a built wheel
prints nothing for `--list-targets`, `--list-modalities`, `--list-properties`,
`--list-core` and `--list-settings`. Upstream `install.sh` runs `pip install -e .` for
that reason, and the adapter refuses an empty vocabulary rather than accepting every
spelling.

The AlphaFold parameters live on a Volume. They are one 5.3 GB archive from
storage.googleapis.com, which is a host the compute skill deliberately never seeds into
an allowlist policy, and a Modal sandbox wipes its filesystem on every submit. HYDRATE
runs in the trusted kernel outside the network fence, so the archive is fetched once
into the Volume and every later job reads it offline.

The accelerator extra is named rather than detected. Upstream `install.sh` reads the
CUDA major version from `nvidia-smi`, and an image build has no GPU to ask, so naming
`cuda12` against a CUDA 12.4 base is the deliberate choice. The recorded BindCraft2
bring-up used A100-80GB, which is the default tier here.

BindCraft 2 sets neither `TF_FORCE_UNIFIED_MEMORY` nor `XLA_PYTHON_CLIENT_MEM_FRACTION`
in its own source, read at the pinned revision, so this image needs none of the source
patching that ColabFold requires on Modal. The two values are still set explicitly,
because Modal runs gVisor, which has no unified memory for JAX to fall back on.
"""

import modal

# The revision tool-catalogue.md records as read on 2026-09-21, five commits past the
# v1.0.0 tag. Upstream tagged v1.0.1 at 5342aefa later the same day and nothing here has
# been re-read at that tag, so this image builds what the catalogue row describes.
_BINDCRAFT2_PIN = "18a9042fbe9a5373a5b7d98fe82335127c2fd70d"
_BINDCRAFT2_REPOSITORY = "https://github.com/PacesaLab/BindCraft2.git"
_BINDCRAFT2_CLONE = "/opt/bindcraft2"

# Upstream weight_cache() reads BINDCRAFT_WEIGHTS and falls back to
# ~/.cache/bindcraft, and the AlphaFold parameters sit in its alphafold subdirectory.
# Mounting the cache root rather than that subdirectory keeps both checkpoint sets and
# any later cache entry on the same Volume.
_WEIGHTS_MOUNT = "/weights/bindcraft"
_WEIGHTS_VOLUME_NAME = "claude-binder-bindcraft2-weights"
_JAX_CACHE_MOUNT = "/root/.cache/jax"
_JAX_CACHE_VOLUME_NAME = "claude-binder-bindcraft2-jax-cache"

_ENVIRONMENT = {
    "BINDCRAFT_WEIGHTS": _WEIGHTS_MOUNT,
    # bindcraft/af2.py reads this and skips its compilation cache when it is unset.
    "JAX_COMPILATION_CACHE_DIR": _JAX_CACHE_MOUNT,
    "TF_FORCE_UNIFIED_MEMORY": "0",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "JAX_PLATFORMS": "cuda",
}

META = {
    "packages": ["bindcraft", "jax", "cuequivariance", "numpy", "scipy"],
    # The recorded bring-up used this GPU. Peak memory remains unmeasured.
    "gpu_default": "A100-80GB",
    # Every host a job dials at run time. Nothing: the checkpoints are hydrated into a
    # Volume outside the fence and read offline afterwards. The optional trajectory
    # viewer references 3Dmol.org from the HTML it writes, which the reader's browser
    # fetches rather than the job.
    "egress_domains": [],
}


def build(
    *, secrets: dict[str, str] | None = None
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Build the image and declare the checkpoint and compilation-cache Volumes."""
    del secrets
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-runtime-ubuntu22.04", add_python="3.12"
        )
        .apt_install("git")
        .run_commands(
            f"git clone --filter=blob:none {_BINDCRAFT2_REPOSITORY} {_BINDCRAFT2_CLONE}",
            f"cd {_BINDCRAFT2_CLONE} && git checkout --quiet {_BINDCRAFT2_PIN}",
            f"cd {_BINDCRAFT2_CLONE} && pip install --no-cache-dir -e '.[cuda12]'",
        )
        .env(_ENVIRONMENT)
    )
    volumes = {
        _WEIGHTS_MOUNT: modal.Volume.from_name(
            _WEIGHTS_VOLUME_NAME, create_if_missing=True
        ),
        _JAX_CACHE_MOUNT: modal.Volume.from_name(
            _JAX_CACHE_VOLUME_NAME, create_if_missing=True
        ),
    }
    return image, volumes, dict(_ENVIRONMENT)


# Upstream's own weight placement, which downloads the 5.3 GB AlphaFold archive when the
# Volume lacks it and exits 2 naming any checkpoint that is missing or unfinished. It is
# idempotent, so a rebuild against a hydrated Volume costs one directory read.
HYDRATE = ("bindcraft", "fetch-weights")
