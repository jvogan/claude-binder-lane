"""Pinned Modal environment for the RFdiffusion generator route.

The image carries the RFdiffusion checkout at ``/opt/rfd``. The checkpoint
Volume mounts at ``/weights``. The adapter selects this route with
``--runner-protocol modal`` and does not need a checkout or checkpoint on the
user's machine.

This recipe is intentionally offline at job time. ``HYDRATE`` is the separate
environment setup step that populates the Volume before a job is submitted.
The dispatcher attaches the same named Volume to every generator job and uses
fresh handles between waves, so the close barrier is the commit point and no
reload call is needed in the job body.
"""

import modal


META = {
    "packages": [
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "dgl",
        "e3nn",
        "rdkit",
        "scipy",
        "Bio",
        "hydra",
        "se3_transformer",
        "rfdiffusion",
    ],
    "gpu_default": "A100",
    "supersedes": ["rfdiffusion"],
    # Every host a job dials at run time. An undeclared domain fails at run time
    # under an allowlist policy, and every declaration is ignored under a
    # no-network policy.
    "egress_domains": [
        # The RFdiffusion checkpoints, matching _WEIGHTS_URL below. HYDRATE pulls
        # them into the Volume, and the host stays declared because a cold Volume
        # or a changed revision sends the job back to the source. The field
        # previously named dl.fbaipublicfiles.com, which this recipe never dials.
        "files.ipd.uw.edu",
    ],
}


_RFD_SHA = "2d0c003df46b9db41d119321f15403dec3716cd9"
_WEIGHTS_URL = "https://files.ipd.uw.edu/pub/RFdiffusion/"
_WEIGHTS_VOLUME_NAME = "claude-science-rfdiffusion-weights"


def build(
    *, secrets: dict[str, str] | None = None
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Build the image and declare the checkpoint Volume mount."""
    del secrets
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11"
        )
        .apt_install("git", "wget", "build-essential")
        .env({"CC": "gcc", "CXX": "g++"})
        .pip_install("torch==2.1.2", "numpy~=1.26.4", "pandas~=2.1.4")
        .pip_install(
            "torch-geometric==2.6.1",
            "torch-scatter==2.1.2",
            "torch-sparse==0.6.18",
            "torch-cluster==1.6.3",
            find_links="https://data.pyg.org/whl/torch-2.1.0+cu121.html",
        )
        .pip_install(
            "dgl==1.1.3",
            find_links="https://data.dgl.ai/wheels/cu121/repo.html",
        )
        .pip_install(
            "e3nn==0.5.1",
            "hydra-core==1.3.2",
            "pyrsistent==0.20.0",
            "rdkit==2024.3.5",
            "scipy==1.13.1",
            "biopython==1.79",
            "fair-esm==2.0.0",
            "networkx==3.2.1",
            "pyyaml==6.0.1",
            "prody==2.6.1",
        )
        .run_commands(
            "git init /opt/rfd && cd /opt/rfd && "
            "git remote add origin "
            "https://github.com/RosettaCommons/RFdiffusion.git && "
            f"git fetch --depth 1 origin {_RFD_SHA} && "
            "git checkout FETCH_HEAD",
            "cd /opt/rfd/env/SE3Transformer && pip install --no-deps .",
            "cd /opt/rfd && pip install --no-deps .",
        )
    )
    volumes = {
        "/weights": modal.Volume.from_name(_WEIGHTS_VOLUME_NAME, create_if_missing=True),
    }
    return image, volumes, {}


HYDRATE = (
    "bash",
    "-lc",
    "set -e; mkdir -p /weights; cd /weights; "
    "for f in Base_ckpt.pt Complex_base_ckpt.pt InpaintSeq_ckpt.pt "
    "InpaintSeq_Fold_ckpt.pt ActiveSite_ckpt.pt Base_epoch8_ckpt.pt "
    "Complex_Fold_base_ckpt.pt; do "
    '  python -c "import zipfile,sys; '
    "  sys.exit(0 if zipfile.is_zipfile('$f') else 1)\" 2>/dev/null || "
    f'  {{ rm -f "$f"; wget -qc {_WEIGHTS_URL}"$f"; }}; done',
)
