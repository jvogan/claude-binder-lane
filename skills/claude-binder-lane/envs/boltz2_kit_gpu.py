"""Pinned Modal environment for the accelerated Boltz-2 kit.

This is the first environment in this lane that installs an Anthropic
optimization kit rather than an upstream model on its own. The kit wraps stock
Boltz-2 without editing it: `stock/` holds the pinned upstream wheel, `opt/`
holds the acceleration package, and `run.sh` is the entry point. The image
carries the kit at ``/kit/boltz2`` and the shared core at ``/kit/common/opt_core``,
both installed editable, which is what ``run.sh install`` does.

**Why this recipe exists.** The Claude Science Modal job surface runs on the
user's own account and accepts a user-supplied image. The kit's own Dockerfile
builds on ``python:3.11.12-slim-bookworm`` with no CUDA base image, because
torch 2.12.0 carries the CUDA 13.0 runtime in its nvidia wheels. Nothing is
compiled for a GPU at build time, and the kit's Dockerfile states that one image
serves H100 and A100. That makes the kit expressible as an ordinary Modal image,
so an accelerated build is reachable from this package without a new route class.

**The mode is the whole point, and it is checked rather than assumed.** The kit
runs in one of four modes. ``off`` is stock as published, ``exact`` reproduces
the unmodified model, ``fast`` is the package default, and ``big`` lowers peak
memory. The kit prints ``[boltz2-opt] ACTIVE mode=<mode>`` when it engages and
``[boltz2-opt] NOT ACTIVE: <reason>`` with exit 3 when it cannot. A caller that
does not read that line cannot tell an accelerated run from a stock one, so the
adapter binding this environment asserts it.

**``run.sh check`` needs a GPU, and it needs the cache.** This docstring first
called it a free pre-flight that engages a mode without a GPU job. That was
wrong on three counts, and the wrong version would have had someone plan a
cost-free verification step that cannot run. ``configs/<card>.env`` builds
``MODEL_OPT_STACK_KEY`` by calling ``boltz2_opt.modes.jit_cache_key``, whose key
carries the compute capability of a real CUDA device, and a failed probe exits 3
by name rather than falling through to an unknown key. The same file states that
``BOLTZ_CACHE`` has no default and that every route refuses by name before any
launch when it is unset or missing a file, so ``check`` also fails before
``HYDRATE`` has run. The kit's own README calls the dry run "GPU, pins, cache,
the mode's levers", and both its setup routes attach a device, with ``--gpus
all`` under Docker and ``--nv`` under Apptainer. So ``CHECK`` is a GPU-attached
dry run that runs no prediction, and it is ordered after ``HYDRATE``. It is the
cheapest place to find a silent fall back to stock, and it is not free.

**What is not settled here.** The kit's Dockerfile states the host needs an
NVIDIA driver that runs CUDA 13.0, which it gives as 580 or newer. This package
has not established which driver the Modal job surface presents, and no file in
this repo records one. TODO(evidence): confirm the driver version a Modal GPU
worker reports before a paid run, because a driver below 580 fails at CUDA
initialisation rather than falling back to stock.

**The image is large, and the first run pays a compile cost.** An earlier version
of this paragraph got the size, the coverage and the reason wrong, so here is
what was measured from the commit with `git ls-tree -r -l` on 2026-09-18.
`common/opt_core` is 316.8 MB, its `kernels/` subtree is 307.2 MB, and the
compiled artefacts inside it, the `.so` and `.cubin` files, are 188.8 MB across
161 files. One 18.2 MB JSON cell table accounts for much of the difference, so
the shared core is not 299 MB of kernels.

Prebuilt binaries cover `sm_80` and `sm_90`/`sm_90a`, which are A100 and H100.
They do not cover every lever. Of the kernel families under `kernels/`, only
`triattn` carries a build keyed to this kit's pinned `torch 2.12.0+cu130`.
`trimul/tx_sm90a`, `transition/flash_sm90a`, `transition/flash_prebuilt` and
`apb/dit_exact` carry no 2.12.0 key at all.

The reason nothing compiles at build time is not that everything is prebuilt. The
kit's own `Dockerfile` and `requirements.lock` both say Triton builds the kits'
kernels and their launchers **at first use**, which is why `build-essential` is
installed in the image rather than only at build time. So a cold container
compiles before it predicts.

**That cost is repaid per container unless the cache is given a home.** The kit's
`configs/<card>.env` documents `MODEL_OPT_JIT_ROOT` as a persistent cache root,
keyed under it by the running torch, CUDA and architecture. Unset, the cache
lands in a temporary directory that a Modal container discards, so every cold
start recompiles. This recipe points it at a directory on the weights Volume,
which survives between jobs. The kit ships an optional pre-filled cache layer
for the same purpose and the public tree carries no such archive, so that layer
is a no-op here and the Volume is the route that works.

**No weights are in the image.** ``HYDRATE`` fills the Volume through the kit's
own weights step, which calls upstream's downloader and then checks each file
against the digest in ``stock/PINS.json``. ``BOLTZ_CACHE`` names the mount at
run time, which is the variable upstream's own ``get_cache_path`` reads.

Sources for every pin below, all read at the release commit
``f4f62fa6`` of ``github.com/anthropics/uplifting-biomolecular-modeling``:
``boltz2/environment/Dockerfile`` for the base image, the apt packages, the
install order and the process environment; ``boltz2/environment/requirements.lock``
for the stack; ``boltz2/stock/PINS.json`` for the upstream commit, the wheel and
the four weight digests; ``boltz2/run.sh`` for the subcommands and the exit
codes. See `accelerated upstream builds
<../references/accelerated-upstream-builds.md>`_.
"""

import modal


META = {
    "packages": [
        "torch",
        "triton",
        "cuequivariance",
        "pytorch_lightning",
        "numpy",
        "rdkit",
        "scipy",
        "boltz",
        "boltz2_opt",
        "opt_core",
    ],
    # PINS.json records "tested_on": "NVIDIA H100 80GB HBM3 with this stack",
    # and the kit ships configs/h100.env beside configs/a100.env and
    # configs/h200.env. H100 is the configuration the published speed-ups were
    # measured on, so it is the tier this recipe sizes for. A100-80GB is a
    # supported fall-back the kit configures, and it is not what any published
    # number describes.
    "gpu_default": "H100",
    # This environment supersedes nothing. `proteomics_boltz_gpu` is the
    # platform's own unmodified Boltz environment and stays as it is, because a
    # campaign comparing accelerated against stock needs both.
    "supersedes": [],
    # Hosts a job or HYDRATE dials, which is what this field declares. Under an
    # allowlist policy a job reaches only the merged list, so a build-time host
    # does not belong here even though the build needs one. This list first
    # carried `github.com`, `codeload.github.com`, `pypi.org` and
    # `files.pythonhosted.org`, which the image build dials and no job does.
    # They are named in `build` instead, where they are used.
    #
    # The four below are exactly the row this package already records for the
    # platform's unmodified Boltz environment, `proteomics_boltz_gpu` in
    # `claude_binder/adapters/modal_platform.py`. The parity is the point,
    # because the kit fills its cache with upstream's own downloader, so it
    # dials what upstream dials. An earlier version of this comment claimed that
    # parity while listing three of the four.
    "egress_domains": [
        # The Boltz-2 cache, pulled by upstream's downloader during HYDRATE.
        "model-gateway.boltz.bio",
        "huggingface.co",
        "*.hf.co",
        # The MSA server. The kit's own `cli_defaults` in stock/PINS.json set
        # `use_msa_server` false and `msa_server_url` to this host, so a default
        # run never dials it. It stays declared because the flag is a run-time
        # choice and an undeclared host fails at run time under an allowlist.
        "api.colabfold.com",
    ],
}


# github.com/anthropics/uplifting-biomolecular-modeling, the single commit on
# main, "Initial public release of the model-optimization kits", Apache-2.0.
#
# The whole SHA, not the abbreviation the release notes and this package's prose
# use. `git fetch --depth 1 origin f4f62fa6` answers "couldn't find remote ref"
# against this remote: a fetch by object name needs the full forty characters.
# Checked by running both forms on 2026-09-18.
_KIT_SHA = "f4f62fa6592ae4938d49b1757bea0cfeff9f468e"
_KIT_REPO = "https://github.com/anthropics/uplifting-biomolecular-modeling.git"
# stock/PINS.json: upstream.commit, and upstream.wheel.file beside it. The kit
# installs the bundled wheel rather than resolving boltz from the index, so this
# commit identifies the stock code the acceleration wraps.
_BOLTZ_SHA = "cb04aeccdd480fd4db707f0bbafde538397fa2ac"
_BOLTZ_VERSION = "2.2.1"
# stock/PINS.json, weights.files. HYDRATE checks each downloaded file against
# these, and the kit's own weights step exits nonzero on a mismatch.
_WEIGHT_DIGESTS = {
    "boltz2_conf.ckpt": "090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1",
    "boltz2_aff.ckpt": "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e",
    "ccd.pkl": "2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4",
    "mols.tar": "39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7",
}
_WEIGHTS_VOLUME_NAME = "claude-science-boltz2-kit-weights"
_KIT_ROOT = "/kit"
_WEIGHTS_MOUNT = "/weights"
# On the Volume on purpose, so the Triton cache survives a cold container.
_JIT_CACHE_ROOT = f"{_WEIGHTS_MOUNT}/jit"


def build(
    *, secrets: dict[str, str] | None = None
) -> tuple["modal.Image", dict[str, "modal.Volume"], dict[str, str]]:
    """Build the kit image and declare the Boltz-2 cache Volume mount.

    Every step here is the kit's own Dockerfile in Modal's builder form, in the
    same order and with the same flags. The one deliberate difference is how the
    kit tree arrives. The Dockerfile copies it from a local build context, which
    Modal has no equivalent of, so this clones the release commit instead and
    takes only the two directories the Dockerfile copies.

    The build dials four hosts that no job dials: ``github.com`` and
    ``codeload.github.com`` for the kit bundle, and ``pypi.org`` and
    ``files.pythonhosted.org`` for the pinned stack, including torch's CUDA 13
    runtime as ``nvidia-*-cu13`` wheels. They are named here rather than in
    ``META["egress_domains"]``, which declares what a running job reaches.
    """
    del secrets
    image = (
        # The kit's own base. No CUDA base image on purpose: torch 2.12.0 brings
        # the CUDA 13.0 runtime, cuBLAS, cuDNN 9.20 and NCCL 2.29 as wheels, so
        # a CUDA base would only add a second runtime to disagree with.
        modal.Image.from_registry("python:3.11.12-slim-bookworm")
        # build-essential because Triton compiles the kit's kernels and their
        # launchers at first use, and gfortran because the tested stack carried
        # it. git is this recipe's own addition, for the clone below.
        .apt_install("build-essential", "gfortran", "git")
        .run_commands(
            # One string, so one layer. Each `run_commands` entry is its own
            # layer and a file deleted in a later one still occupies the earlier
            # one, so cloning in one entry and cleaning up in the next kept the
            # 318 MiB pack and a second copy of the worktree in the image. That
            # was about 930 MiB of layers for a 306 MiB payload. `mv` rather
            # than `cp -a` for the same reason.
            #
            # The two paths are the ones the kit Dockerfile copies, and nothing
            # else. The leading slash anchors each pattern at the repository
            # root. Without it, `boltz2/` and `common/` match a directory of
            # that name at any depth, and the checkout also pulls af3_jax,
            # atlasfold and colabdesign. Checked by running both patterns on
            # 2026-09-18.
            f"git init {_KIT_ROOT}_src && cd {_KIT_ROOT}_src && "
            f"git remote add origin {_KIT_REPO} && "
            "git config core.sparseCheckout true && "
            "printf '/boltz2/\\n/common/opt_core/\\n' > .git/info/sparse-checkout && "
            f"git fetch --depth 1 origin {_KIT_SHA} && "
            "git checkout FETCH_HEAD && "
            f"mkdir -p {_KIT_ROOT}/common && "
            f"mv {_KIT_ROOT}_src/boltz2 {_KIT_ROOT}/boltz2 && "
            f"mv {_KIT_ROOT}_src/common/opt_core {_KIT_ROOT}/common/opt_core && "
            f"cd / && rm -rf {_KIT_ROOT}_src",
        )
        .run_commands(
            # The Dockerfile's install order, kept exactly. pip and wheel at the
            # lock's versions first, then every distribution at the lock's
            # version with --no-deps because the lock is complete, then the
            # bundled stock wheel. The setuptools build constraint is the
            # release the source-only distributions in the lock were built with.
            f"cd {_KIT_ROOT}/boltz2 && "
            "grep -E '^(pip|wheel)==' environment/requirements.lock > /tmp/pip.txt && "
            "python3.11 -m pip install --no-cache-dir --no-deps -r /tmp/pip.txt",
            f"cd {_KIT_ROOT}/boltz2 && "
            "printf 'setuptools==82.0.1\\n' > /tmp/build-constraints.txt && "
            "grep -v -E '^(#|pip==|wheel==|boltz==)' environment/requirements.lock > /tmp/stack.txt && "
            "python -m pip install --no-cache-dir --no-deps "
            "--build-constraint /tmp/build-constraints.txt -r /tmp/stack.txt",
            f"cd {_KIT_ROOT}/boltz2 && "
            "python -m pip install --no-cache-dir --no-deps "
            f"stock/boltz-{_BOLTZ_VERSION}-py3-none-any.whl",
            "rm -f /tmp/stack.txt /tmp/pip.txt /tmp/build-constraints.txt",
        )
        .run_commands(
            # The one install step. It installs the shared core and the kit
            # package editable, then runs stock/check_pins.py, which exits 3 if
            # boltz is not installed at the pin. A build that reaches the next
            # layer has passed that check.
            f"cd {_KIT_ROOT}/boltz2 && bash run.sh install",
        )
        .env(
            {
                # The process environment of the tested stack, from the
                # Dockerfile's final ENV. PYTHONHASHSEED is fixed, Triton's
                # launchers carry no debug info, and the CUDA 13 libraries in
                # the nvidia wheels are on the search path for extensions that
                # do not carry their own.
                "PYTHONHASHSEED": "0",
                "CFLAGS": "-g0",
                "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib",
                # Upstream's own get_cache_path reads this. The Dockerfile
                # deliberately leaves it unset and sets it at run time; this
                # recipe sets it because the mount point below is fixed.
                "BOLTZ_CACHE": _WEIGHTS_MOUNT,
                # Triton compiles this kit's kernels at first use. Unset, its
                # cache goes to a temporary directory the container discards,
                # so every cold start recompiles. The Volume outlives the
                # container, and the kit keys the cache by torch, CUDA and
                # architecture underneath this root, so one directory serves
                # every card without mixing builds.
                "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
            }
        )
    )
    volumes = {
        _WEIGHTS_MOUNT: modal.Volume.from_name(
            _WEIGHTS_VOLUME_NAME, create_if_missing=True
        ),
    }
    return image, volumes, {
        "BOLTZ_CACHE": _WEIGHTS_MOUNT,
        "MODEL_OPT_JIT_ROOT": _JIT_CACHE_ROOT,
    }


# The kit's own weights step. It calls upstream's downloader into the directory
# and then checks the cache and every digest, so a partial or corrupted download
# fails here rather than during a paid prediction. run.sh install --weights DIR
# is the documented form (README.md Setup, run.sh line 3).
HYDRATE = (
    "bash",
    "-lc",
    f"set -e; mkdir -p {_WEIGHTS_MOUNT} {_JIT_CACHE_ROOT}; cd {_KIT_ROOT}/boltz2; "
    f"bash run.sh install --weights {_WEIGHTS_MOUNT}",
)


# The card configuration files the kit ships. `configs/a100.env` states that it
# mirrors `configs/h100.env` and that the two differ only in the target label and
# their header comments, so a wrong card here is reported rather than refused.
# h200 is absent from this package's GPU tiers, so it is listed and unreachable.
KIT_CARDS = ("h100", "a100", "h200")
KIT_MODES = ("off", "exact", "fast", "big")


def check_command(*, card: str = "h100", mode: str = "exact") -> tuple[str, ...]:
    """Return the kit's dry run for one card and mode.

    It resolves the mode, gates it, and plans the partials without applying
    anything and without predicting anything. Exit 3 means the mode would not
    engage, and the reason is on the NOT ACTIVE line.

    **This needs a GPU attached and the cache already hydrated.** It is cheap
    next to a prediction and it is not free, so run it after ``HYDRATE`` on the
    same worker, never as a pre-flight on a machine with no device.

    **Pass the card you are actually running on.** This was a constant with
    ``h100`` and ``exact`` baked in, which is wrong twice over on any other card.
    The card matters because `configs/<card>.env` sets ``MODEL_OPT_TARGET_GPU``
    and a mismatch is reported. The mode matters more. On A100 the kit's own
    documentation records that ``exact`` drops ``msa_pwa_exact`` as a declared
    ``card_off`` lever, that ``fpf_trimul_exact`` idles and serves the bytes of
    ``--mode off``, and that it exits 3 below 64 tokens, between 1,024 and 1,099
    tokens, and in narrow bands above. ``fast`` and ``big`` run on A100 without
    those caveats. So a check hardcoded to ``exact`` gates the one mode least
    likely to engage on the more commonly available card.
    """
    if card not in KIT_CARDS:
        raise ValueError(f"{card} is not a card this kit configures: {KIT_CARDS}")
    if mode not in KIT_MODES:
        raise ValueError(f"{mode} is not a kit mode: {KIT_MODES}")
    return (
        "bash",
        "-lc",
        f"cd {_KIT_ROOT}/boltz2 && bash run.sh check --config {card} --mode {mode}",
    )


# The default pairing, which is the card and the mode the published figures were
# measured on. Anything else goes through `check_command`.
CHECK = check_command()
