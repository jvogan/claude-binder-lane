"""Pinned Modal environment for the Genie3 binder-mode generator route.

The image carries the Genie3 checkout at ``/opt/genie3``. The weights Volume
mounts at ``/weights``, and ``GENIE3_HOME`` names it. Genie3 opens
``pretrained/v1/config.yaml`` relative to the process working directory, so the
adapter has to start the subprocess with ``cwd`` set to ``/weights``. A home
directory without that file is the failure that exits in under a second with a
FileNotFoundError.

This recipe is offline at job time. The source is baked into the image at a
pinned commit and ``HYDRATE`` populates the weights Volume before any job is
submitted. Nothing in the job body reaches the network.

**This recipe covers ``genie3 generate`` only, and deliberately omits the
evaluation stack.** It skips upstream ``scripts/setup/setup.sh``, so it carries
no ColabFold, no AlphaFold2 multimer parameters, no ESMFold, no FoldSeek, no DSSP
and no TMscore. That install is roughly 30 minutes and 30 GB, and every piece of
it serves ``genie3 evaluate``.

What the omission costs, stated plainly: **this environment cannot run
``genie3 evaluate``, and it cannot run ``genie3 run``, which chains generation
into evaluation.** ``genie3 generate`` writes backbone structures and nothing
else, so sequences have to come from the package's own sequence-design stage.
That is the route the package takes anyway. A served binder-mode generation at
n_sample 1 wrote one PDB and zero sequence files, which is the shape the
downstream designer expects.

There is no CUDA base image here, unlike ``rfdiffusion_generator_gpu``, and that
is on purpose rather than an omission. A base image pinned to one CUDA release
and a torch wheel built against another is the trap worth avoiding, because the
mismatched wheel installs quietly instead of failing. This recipe removes the
mismatch by having no CUDA base to mismatch: the torch wheel supplies the CUDA
runtime through its own pinned nvidia cu12 dependencies. The measured binder-mode
environment built exactly this way and reported CUDA runtime 12.6 to match its
cu126 torch wheel.
"""

import modal


META = {
    "packages": [
        "torch",
        "numpy",
        "huggingface_hub",
        "genie3",
    ],
    # A100, at 40 GB. This lane's dispatcher accepts four tiers and no others:
    # A10G at 24 GB, A100 at 40, A100-80GB at 80, H100 at 80. That table is
    # GPU_TIERS in scripts/dispatch_modal.py, and gpu_tier_for reads the value
    # below straight out of SHIPPED_GPU_DEFAULTS. A card outside the four cannot
    # be requested, so gpu_default has to name one of them.
    #
    # The basis is the declared requirement plus headroom, and it was never
    # measured. catalog.json declares gpu_required true and gpu_memory_gb 24,
    # and that 24 descends from the profile template's resource block rather
    # than from any Genie3 memory measurement. No such measurement exists in
    # this package.
    #
    # A100 clears the declared 24 GB with headroom. The package has no
    # attributable peak-memory measurement for this tool.
    #
    # A10G was rejected. It meets the declaration exactly at 24 GB and leaves
    # nothing spare, and it sits below every card Genie3 has been observed to
    # run on. The contract's gpu_memory_gb is that same 24, so gpu_tier_for
    # would never escalate off A10G, and a requirement the declaration
    # understates would surface as an OOM on hardware nobody measured.
    #
    # The port source's own VRAM figure for length 280, 18 to 26 GB, is an
    # extrapolation from Genie 2 on an A6000 and its note says so. It is not used
    # here. Binder mode generates 60 to 90 residue binders, well under 280.
    "gpu_default": "A100",
    # Claude Science ships no bundled Genie3 environment for this to replace.
    "supersedes": [],
    # Every host a job dials at run time. An undeclared domain fails at run time
    # under an allowlist policy, and every declaration is ignored under a
    # no-network policy.
    #
    # A job dials nothing. The weights host stays declared because a cold Volume
    # or a changed revision sends HYDRATE back to the source.
    #
    # Two entries, not one. The Hub serves metadata from huggingface.co and
    # redirects LFS objects to a CDN under hf.co. Every bundled environment in
    # this package that reads from the Hub declares the pair, in
    # claude_binder/adapters/modal_platform.py for proteomics_gpu and
    # proteomics_boltz_gpu.
    "egress_domains": [
        "huggingface.co",
        "*.hf.co",
    ],
}


# aqlaboratory/genie3. The commit is recorded in the seed lane's install record
# and pinned again as GENIE3_SOURCE_COMMIT in that lane's fal application. Both
# were verified against the upstream repository on 2026-09-11: the commit exists
# and its message is "Update README.md".
_GENIE3_SOURCE_SHA = "5214459c42e69b01fadfc7d7ebda09d8e5082115"
_WEIGHTS_REPO_ID = "yeqinglin/genie3"
_WEIGHTS_REVISION = "9ae31ebb8c56eebdc05ab282a8fd3f6a6d2a03a2"
# Genie3 publishes a directory of weights, not one checkpoint file. This is the
# subtree the install recipe, the RunPod bootstrap, and the fal application all
# fetch, and it is the only part of the repository the generator reads. The
# training data under data/train/** stays out.
_WEIGHTS_INCLUDE = "pretrained/**"
_WEIGHTS_VOLUME_NAME = "claude-science-genie3-weights"
_WEIGHTS_MOUNT = "/weights"
_MODEL_VERSION = "v1"

# Upstream setup.py hard-pins torch==2.7.1. Three port-source records state that
# pin: the install recipe's trap list, its source list, and the comment above the
# requirements block of the shipped fal application. No genie3 source checkout
# survives on this machine, so setup.py was not read first-hand.
#
# The index is the part the records disagree on, and a measurement settles it.
# The install recipe says to install 2.7.1 from the cu124 index. The run log for
# that same recipe records the attempt failing, because the cu124 index carries
# nothing above 2.6.0+cu124. The operator patched setup.py to accept
# torch>=2.5.0,<3 and installed 2.6.0+cu124 to get moving. That is a workaround
# for a wrong index, not evidence against the 2.7.1 pin.
#
# cu126 is what a working binder-mode environment actually reports. A served
# Genie3 binder-mode run recorded torch_version 2.7.1+cu126 and
# cuda_runtime_version 12.6 on python 3.12, in an environment whose only torch
# requirement was the bare `torch==2.7.1`. See _MEASURED_RECEIPT below.
#
# So the pair `torch==2.7.1` from cu124 cannot be written here. The two halves
# contradict each other and pip resolves the contradiction by installing
# something else, which is the silent-wrong-build failure this pin exists to
# prevent. cu126 is the index that carries the pinned version, and it is the
# build a working environment reported. There is no base image CUDA release for
# it to disagree with, because this recipe pins no CUDA base image.
_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"
_TORCH_VERSION = "2.7.1"
# torchvision is deliberately absent, and it is left out because it could not be
# sourced as a `genie3 generate` requirement. tool-catalogue.md lists
# `torchvision==0.22.1 --no-deps` as a Genie3 environment constraint, and the
# port source does pin exactly that. Every place it pins it is a binder-hunt
# bridge manifest, where the comment states the purpose: Boltz shares that pod
# and pulls a torchvision incompatible with the image's torch 2.7.1+cu126. That
# is a Boltz repair on a shared pod, not something generation reads. The measured
# binder-mode environment installed no torchvision and generated a backbone.
#
# The catalogue's `numpy<2.2` row reads the same way. The port source pins it
# immediately before invoking Boltz, because Boltz's numba dependency caps numpy
# at 2.1. It belongs to that handoff, not to this environment.
#
# setup.py's own stated pins, from the install recipe's source list, are
# numpy>=2.0.2,<3 and biopython<1.86. numpy and Cython go in before the source
# install, the way the recipe orders them. setup.py resolves biopython itself.
_NUMPY_SPEC = "numpy>=2.0.2,<3"
# The RunPod install used a py3.11 base image. The measured binder-mode
# environment ran python 3.12.
_PYTHON_VERSION = "3.12"

# The served binder-mode run this recipe reproduces. Read from the fal receipt of
# a completed smoke attempt of the seed lane's `generate-genie3` stage. It is the
# only measurement of a complete Genie3 binder-mode environment in either
# repository. Every field here was read, not modeled.
#
# That attempt generated, it did not only probe. The configuration beside the
# receipt sets generation.dataset.source to target with n_sample 1, carries no
# evaluation block, and the attempt wrote
# runs/binder-lane/pdbs/binder-lane_0.pdb. So one N=1 binder-mode backbone came
# out of an environment holding torch and the Genie3 source and nothing else.
# The receipt's own environment identity string is not copied here. It embeds the
# name of a private deployment, and no value in this recipe depends on it. The
# revisions it carries are the two pinned above.
_MEASURED_RECEIPT = {
    "torch_version": "2.7.1+cu126",
    "cuda_runtime_version": "12.6",
    "device": "cuda NVIDIA H100 80GB HBM3",
    "weights_file_count": 4,
    "checkpoint_bytes": 536135478,
    "runner_wall_seconds": 58.816,
}
# The weights subtree the measurement observed. HYDRATE holds the tree it
# downloads against both numbers, because a truncated download that still loads
# is the failure that produces a plausible wrong backbone count later.
_WEIGHTS_FILE_COUNT = _MEASURED_RECEIPT["weights_file_count"]
_WEIGHTS_TOTAL_BYTES = _MEASURED_RECEIPT["checkpoint_bytes"]

# Upstream scripts/setup/setup.sh installs ColabFold, AlphaFold2 multimer
# parameters, ESMFold, FoldSeek, DSSP and TMscore. The install recipe calls that
# stack a 30 minute, 30 GB install needed only for `genie3 evaluate`, and says to
# skip it. That recipe describes unconditional mode, so the question is whether
# the binder path needs it back.
#
# It does not, because the package never runs `genie3 evaluate`. It runs
# `genie3 generate` and hands the backbones to its own sequence designer. The
# binder-mode config synthesizer does write an evaluation block naming colabfold,
# but generation never reads it, so the block is inert.
#
# Four records agree, three of them stated and one measured. The seed lane's
# binder hunt runs `genie3 generate` and its comment states that `run` would
# evaluate and that evaluation needs a package upstream setup.py does not
# resolve. The seed lane's Genie3 adapter keeps the evaluation section out of the
# configuration by default and says so in its own --evaluation-folding-model help
# text. Its fal application writes a configuration with no evaluation block at
# all. And the measured attempt wrote one PDB and zero sequence files, which is
# what a generate-only route should produce.
#
# GENIE3_ALLOW_COLABFOLD_PARAMS=1 therefore has no place in this recipe. That
# flag gates the port source's RunPod bootstrap before it runs setup.sh. This
# image never runs setup.sh, so it downloads no AlphaFold2 multimer parameters
# and needs no acknowledgement of their terms.
#
# An adapter reading this flag must not pass an evaluation folding model to a job
# in this environment. The evaluation stack is not here, so `genie3 run` and any
# configuration that reaches evaluation would fail on a missing dependency.
_EVALUATION_STACK_SKIPPED = True

_ENV = {
    # The directory that holds pretrained/. The adapter resolves this, and it is
    # also the cwd the generation subprocess must receive.
    "GENIE3_HOME": _WEIGHTS_MOUNT,
}


def build(
    *, secrets: dict[str, str] | None = None
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Build the image and declare the weights Volume mount."""
    del secrets
    image = (
        modal.Image.debian_slim(python_version=_PYTHON_VERSION)
        .apt_install("git", "ca-certificates", "build-essential")
        .run_commands("python -m pip install --upgrade pip setuptools wheel")
        .pip_install(_NUMPY_SPEC, "Cython")
        # torch lands before the source install, so the source build finds its
        # hard pin already satisfied and never resolves a CPU wheel. The install
        # recipe's trap list calls this ordering out.
        .pip_install(f"torch=={_TORCH_VERSION}", index_url=_TORCH_INDEX_URL)
        # The hf binary that HYDRATE calls ships with the [cli] extra from
        # huggingface_hub 0.26 onward. Older releases ship huggingface-cli only.
        .pip_install("huggingface_hub[cli]")
        .run_commands(
            "git init /opt/genie3 && cd /opt/genie3 && "
            "git remote add origin https://github.com/aqlaboratory/genie3.git && "
            f"git fetch --depth 1 origin {_GENIE3_SOURCE_SHA} && "
            "git checkout FETCH_HEAD",
            # --no-build-isolation keeps the build backend from pulling its own
            # torch into a throwaway environment, which is how the install recipe
            # runs it.
            "cd /opt/genie3 && python -m pip install --no-build-isolation .",
        )
        .env(_ENV)
    )
    volumes = {
        _WEIGHTS_MOUNT: modal.Volume.from_name(
            _WEIGHTS_VOLUME_NAME, create_if_missing=True
        ),
    }
    return image, volumes, dict(_ENV)


# Runs once in a sandbox when build_env(hydrate=True) is called, before any job is
# submitted. The download command is the one the install recipe and the RunPod
# bootstrap both run, and the install recipe records it as what upstream
# scripts/setup/download.sh runs too.
#
# Hydration then refuses three failures rather than passing them to a paid job.
# A missing model configuration exits the job in under a second on a
# FileNotFoundError. A wrong file count or byte total means a truncated or
# drifted download, which still loads and produces a plausible wrong result. Both
# numbers come from the measured receipt, so a disagreement means the Volume does
# not hold the tree the campaign scored against.
#
# The verifier runs through a quoted heredoc, so bash passes the program to the
# interpreter with its real newlines and expands nothing inside it.
_HYDRATE_VERIFY = f"""import pathlib, sys

root = pathlib.Path({_WEIGHTS_MOUNT!r}) / "pretrained"
config = root / {_MODEL_VERSION!r} / "config.yaml"
if not config.is_file():
    sys.exit("genie3_generator_gpu: hydration wrote no %s" % config)
files = sorted(path for path in root.rglob("*") if path.is_file())
total = sum(path.stat().st_size for path in files)
if len(files) != {_WEIGHTS_FILE_COUNT} or total != {_WEIGHTS_TOTAL_BYTES}:
    sys.exit(
        "genie3_generator_gpu: the hydrated weights are %d files and %d bytes. "
        "{_WEIGHTS_REPO_ID}@{_WEIGHTS_REVISION} measured "
        "{_WEIGHTS_FILE_COUNT} files and {_WEIGHTS_TOTAL_BYTES} bytes"
        % (len(files), total)
    )
"""

HYDRATE = (
    "bash",
    "-lc",
    f"set -e\n"
    f"mkdir -p {_WEIGHTS_MOUNT}\n"
    f"cd {_WEIGHTS_MOUNT}\n"
    f"hf download {_WEIGHTS_REPO_ID} --revision {_WEIGHTS_REVISION} "
    f"--include '{_WEIGHTS_INCLUDE}' --local-dir .\n"
    "python - <<'GENIE3_HYDRATE_VERIFY'\n"
    f"{_HYDRATE_VERIFY}"
    "GENIE3_HYDRATE_VERIFY\n",
)
