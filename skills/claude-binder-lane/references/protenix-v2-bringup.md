# Protenix v2: reproducible self-hosted bring-up

This is the self-hosted, non-provider-dispatch design for the `protenix-v2`
adapter. It records only first-party Protenix sources, checked on 2026-08-29.
No image build, model download, provider dispatch, credential use, or paid
computation stands behind it, so it establishes no qualified GPU run.

The route is viable in principle, but is not ready to promote from
`operator_configured` until its official weight object can be fetched once and
recorded in the immutable artifact manifest described below. The official
checkpoint URL returned HTTP 403 for both `HEAD` and a one-byte `Range` request
on 2026-08-29. Do not replace it with a third-party Hugging Face checkpoint
merely to make the route appear available.

Re-tested 2026-09-07 and unchanged. `HEAD` returns `403 Forbidden` from
`Server: TosServer`, and the one-byte `Range` request returns a 217-byte JSON body
reading `{"Code":"AccessDenied","Message":"Access Denied","EC":"0003-00000015",
"DetailErrCode":14006}`. The refusal is the object store denying access to the object,
not a network path failure, a redirect, or a rate limit. Re-tested again on 2026-09-13 and
still `403` from `Server: TosServer`, with request id
`1f1c07a6f1e88f446aa6f1e8-bedb802-1x5pNY-HO-cb-tos-1az-front-azc-16`. Fifteen days of
persistence across three checks, and an `AccessDenied` code rather than a transient one, mean
this is unlikely to clear by waiting. Ask upstream for a fetchable object, or leave the route
unpromoted.

The denial is scoped to this caller. A `403` says the object store refused this request, and it
is not evidence that the object is withdrawn or that another network gets the same answer. That
is why the catalogue holds Protenix v2 at `operator_configured` rather than `unbound`. An
operator who can fetch the object records its SHA-256 and the route promotes; nothing else about
the arm is blocked.

**One object of nineteen is denied, measured 2026-09-14.** Earlier checks tested the v2
checkpoint alone, which left the rest of the route's objects untested and the page reading as
though the host were closed. Every object in `protenix.web_service.dependency_url.URL` at the
pinned commit was probed with a one-byte `Range` request. Eighteen answered `206 Partial
Content` with a byte count and an `ETag`. Only `protenix-v2` answered `403`. The four common
inference files this route must hydrate are among the eighteen: `components.cif` at
490,777,362 bytes, `components.cif.rdkit_mol.pkl` at 142,498,117, `clusters-by-entity-40.txt`
at 21,699,572, and `obsolete_release_date.csv` at 134,716. Both template extras answered as
well. So does every other published checkpoint, including `protenix_base_default_v1.0.0` at
1,475,950,125 bytes and `protenix_mini_default_v0.5.0` at 537,049,294.

Reproduce the table by probing each URL in that module with `curl -r 0-0 -o /dev/null -D -`. No
model bytes were downloaded. An `ETag` is the object store's own identifier, not a content
SHA-256, so it supports a retrieval receipt and does not replace the digest this route requires.

**Upstream says why.** On 2026-04-09 a ByteDance maintainer answered the 403 report on
`bytedance/Protenix` issue 296: the accessibility of the protenix-v2 checkpoint is under review
as part of a company-level internal evaluation process, with no timeline offered. Issue 298
states independently that the v2 URL returns 403 while other checkpoints are accessible, which
is what the 2026-09-14 probe measures. Four issues reporting this remain open, opened between
2026-04-08 and 2026-08-19, and two more were closed without the object becoming reachable.
Do not plan on the object clearing, and do not re-test the same URL.

## Where to get a checkpoint

Three routes exist, and each one supports a different claim.

**1. The official v2 object.** `checkpoint/protenix-v2.pt` on the upstream host. This is the only
route that supports a published-reproduction claim, because it is the object the published stack
names. It answers 403 to this workspace and has answered 403 on every check since 2026-08-29. An
operator whose network reaches it fetches it once, records the digest through the hash-checked
route below, and the binding promotes.

**2. A different official checkpoint.** Eighteen objects answer, including
`protenix_base_default_v1.0.0` at 1,475,950,125 bytes and `protenix_mini_default_v0.5.0` at
537,049,294. Fetch one, hash what you fetched, and you have a recordable digest today with no
provenance question at all. This is a substitution: it runs Protenix, and it is a different model
from the v2 the published roster names, so it belongs in the substitution path rather than in a
fidelity claim. The shipped adapter fixes `--model_name protenix-v2`, so selecting another
checkpoint is a code change rather than a configuration change.

**3. A community copy of v2.** `TMF001/protenix-v2-weights` on Hugging Face, revision
`0b3bf48266effd548f3d399e8e76a87def9e9ec4`, one checkpoint file of 1,859,785,497 bytes with
SHA-256 `8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599`. Its model card names
the official upstream URL as the source and declares Apache-2.0, which matches the licence
upstream publishes for its own weights. A commenter named this mirror in upstream issue 332 on
2026-08-31, which is where the route comes from. No maintainer endorsed it there.

*What is established.* The file loads into Protenix 2.0.0 and predicts. Measured on an H100, the
loaded state dict carries the expected module structure, and a 20-residue single chain returned
pLDDT 86.63 with no clash and `prediction complete`. The bytes are a working Protenix v2
checkpoint of the expected architecture.

*What is not established.* Whether these are the weights ByteDance trained and published. Nothing
can settle that while the official object answers 403 and upstream publishes no digest. Checking
your download against the SHA-256 above proves your copy matches this mirror. It says nothing
about whether the mirror matches upstream.

So this route stays operator-supplied. It leaves `WEIGHTS_PROVENANCE_UNVERIFIED` standing at the
licence gate, and it cannot carry a `baseline_fidelity` claim. It is a working way to run
Protenix v2 for design work, substitution studies and demonstrations, chosen with that stated.

Verify any copy before you use it:

```bash
sha256sum protenix-v2.pt
# expect 8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599
```

A mismatch means your download differs from the mirror this page names, and the file should be
discarded rather than loaded.

**A fourth option holds no checkpoint at all.** A hosted provider runs Protenix on its own
weights, so none of the three routes above applies and the 403 never arises. It returns a
prediction and no digest of yours, which is a different kind of result from the ones this page
is built around. [Discover tools and routes](platform-tool-discovery.md#which-catalogue-tools-a-hosted-api-already-serves)
carries the mapping and the authorization rule.

**Egress stays split whichever route you pick.** `huggingface.co` may appear in a hydration-phase
allowlist when an operator chooses route 3, and it must never appear in prediction-time egress.
Prediction runs offline against locally hydrated files on every route.

## Pinned upstream source

Use the upstream `v2.0.0` release, whose peeled Git commit is
`2475421477ab414b571149ad4a875c390ff8a35d`. The tag release was published on
2026-04-07 and the tagged `protenix/version.py` says `2.0.0`. This is the
release whose README introduces `protenix-v2` and whose model configuration
contains the 464.44M-parameter v2 architecture.

Pin source by commit, not by `main`, tag name, PyPI's moving simple-index view,
or a local checkout:

```text
repository: https://github.com/bytedance/Protenix.git
release tag: v2.0.0
commit:      2475421477ab414b571149ad4a875c390ff8a35d
package:     protenix==2.0.0 (only as a cross-check; build from the commit)
```

The source checkout must prove its identity before package installation:

```bash
git clone https://github.com/bytedance/Protenix.git /opt/protenix
git -C /opt/protenix checkout --detach 2475421477ab414b571149ad4a875c390ff8a35d
test "$(git -C /opt/protenix rev-parse HEAD)" = \
  2475421477ab414b571149ad4a875c390ff8a35d
```

## Runtime and image contract

The tagged upstream `Dockerfile` is the authority for the CUDA/Python family:
PyTorch 2.7.1, CUDA 12.6.3, Python 3.11, Ubuntu 22.04, and a development image
with a compiler. Its supplied base image has the following current digest,
resolved from the exact registry named in that Dockerfile:

```text
vemlp-cn-beijing.cr.volces.com/preset-images/pytorch:
  2.7.1-cu12.6.3-py3.11-ubuntu22.04@
  sha256:bb558085abee1d77f0c981220a474d63f55d42143a83580116df67484932c604
```

Pin that digest in the fresh build. It is an `amd64` image. Reject an `arm64`
host image and reject any image that does not report Python 3.11.x,
`torch==2.7.1`, and `torch.version.cuda == "12.6"` before it is used.

The exact upstream direct requirements at the source commit are:

```text
torch==2.7.1                    torchvision==0.22.1
torchaudio==2.7.1               cuequivariance-ops-torch-cu12==0.8.0
cuequivariance-torch==0.8.0     triton==3.3.1
deepspeed==0.17.5               rdkit==2025.9.3
fair-esm==2.0.0                 biopython==1.85
biotite==1.4.0                  modelcif==1.4
gemmi==0.6.7                    pdbeccdutils==1.0.0
ml_collections==1.1.0           tqdm==4.67.1
pandas==2.3.1                   PyYAML==6.0.2
matplotlib==3.10.5              ipywidgets==8.1.7
py3Dmol==2.5.2                  scikit-learn==1.7.1
scikit-learn-extra==0.3.0       protobuf==6.31.1
icecream==2.1.7                 ipdb==0.13.13
wandb==0.21.1                   numpy==2.4.1
optree==0.17.0
```

The upstream requirements file also has three ranges: `scipy>=1.9.0`,
`pydantic>=2.0.0`, and `networkx>=3.4.2`. It is therefore not, by itself, a
reproducibility lock. Before the first image build,
generate a Linux/CPython-3.11 hash lock from the pinned checkout and commit it
beside the image recipe. The resulting lock (including every transitive
dependency and selected wheel hashes) is part of the image identity:

```bash
uv pip compile --python-version 3.11 \
  --python-platform x86_64-manylinux_2_28 \
  --generate-hashes --no-annotate \
  -o protenix-v2-requirements.lock /opt/protenix/requirements.txt
```

Do not subsequently regenerate that lock in a normal build. Install it with
`pip install --require-hashes -r protenix-v2-requirements.lock`, then install
the checked-out Protenix source with `pip install --no-deps /opt/protenix`.
Record the lock SHA-256 and the final image digest in the qualification receipt.

Install the system packages the upstream Dockerfile names:

```text
git g++ gcc libc6-dev make postgresql hmmer kalign
```

Also install NVIDIA CUTLASS because the upstream image and kernel documentation
do. Pin its `v3.5.1` commit rather than using the branch name:

```bash
git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass
git -C /opt/cutlass checkout --detach f7b19de32c5d1f3cedfc735c2849f12b537522ee
export CUTLASS_PATH=/opt/cutlass
```

Protenix compiles its custom LayerNorm kernel when it is first called. Keep
`TORCH_EXTENSIONS_DIR` on a writable persistent volume, not in the disposable
container filesystem. The source's default triangle kernels are cuEquivariance;
the CUTLASS path is also required if an operator selects the `deepspeed` triangle
attention backend. It is not a reason to silently change the adapter's model or
inference parameters.

## Build recipe outline

The eventual image recipe must materially implement the following; it must not
reuse a previous image without rebuilding from these frozen inputs.

```dockerfile
FROM vemlp-cn-beijing.cr.volces.com/preset-images/pytorch@sha256:bb558085abee1d77f0c981220a474d63f55d42143a83580116df67484932c604

RUN apt-get update && apt-get install -y --no-install-recommends \
      git g++ gcc libc6-dev make postgresql hmmer kalign curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/bytedance/Protenix.git /opt/protenix \
 && git -C /opt/protenix checkout --detach 2475421477ab414b571149ad4a875c390ff8a35d \
 && test "$(git -C /opt/protenix rev-parse HEAD)" = 2475421477ab414b571149ad4a875c390ff8a35d

COPY protenix-v2-requirements.lock /tmp/protenix-v2-requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/protenix-v2-requirements.lock \
 && python -m pip install --no-deps /opt/protenix

RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass \
 && git -C /opt/cutlass checkout --detach f7b19de32c5d1f3cedfc735c2849f12b537522ee \
 && test "$(git -C /opt/cutlass rev-parse HEAD)" = f7b19de32c5d1f3cedfc735c2849f12b537522ee

ENV CUTLASS_PATH=/opt/cutlass \
    PROTENIX_ROOT_DIR=/datavol_binder/protenix \
    TORCH_EXTENSIONS_DIR=/datavol_binder/torch_extensions

RUN python -m pip check \
 && protenix --help \
 && python -c 'import sys, torch; assert sys.version_info[:2] == (3, 11); assert torch.__version__ == "2.7.1"; assert torch.version.cuda == "12.6"'
```

The Dockerfile uses a precommitted lock file intentionally. Substitute its
committed SHA-256 in the final recipe/receipt; `COPY` is a placeholder in this
reference, not permission to build with an unreviewed resolver result.

## Official weights and hydration

At the v2 source commit, Protenix itself maps the model name to this first-party
object, not to Hugging Face:

```text
model name: protenix-v2
checkpoint: https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt
local path: ${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt
```

`runner/inference.py` calls `urllib.request.urlretrieve` for a missing
checkpoint, and also pulls four required common files. `HF_HOME` and
`HF_HUB_OFFLINE` do not control this implementation. The hydration volume must
contain the following before an offline prediction starts:

```text
${PROTENIX_ROOT_DIR}/checkpoint/protenix-v2.pt
${PROTENIX_ROOT_DIR}/common/components.cif
${PROTENIX_ROOT_DIR}/common/components.cif.rdkit_mol.pkl
${PROTENIX_ROOT_DIR}/common/clusters-by-entity-40.txt
${PROTENIX_ROOT_DIR}/common/obsolete_release_date.csv
```

If the job sets `--use_template true`, also hydrate the two extra files which
the same source maps under `common/`: `obsolete_to_successor.json` and
`release_date_cache.json`. The binder adapter sends precomputed target MSAs and
does not require template search, RNA MSA search, or the bundled training data.

The checkpoint URL is a mutable pathname and the tagged upstream source does
not publish a content SHA-256, model-object version ID, or signed manifest. Its
filename is thus a *model label*, not an immutable revision. The reproducible
revision strategy is:

1. Fetch only the exact upstream URL into a staging file on the persistent
   volume. Never allow Protenix's auto-download inside a GPU job.
2. Record URL, retrieval UTC time, final redirect URL if any, HTTP status,
   `ETag`, `Last-Modified`, byte count, and the file's SHA-256 in a committed
   `protenix-v2-artifacts.json` receipt.
3. Move the staging file into its required local pathname only after
   `sha256sum -c` succeeds against the value already committed in that receipt.
   Later hydration must refuse a changed byte stream.
4. Hash the four common inference files in the same receipt. Store the source
   commit, requirements-lock hash, base-image digest, CUTLASS commit, model
   SHA-256, and final image digest together as the route's environment identity.

The first successful official retrieval is an explicit evidence-capture event,
not a background image build. While that URL answers 403, no run can receipt the
v2 checkpoint. The four common files and the other published checkpoints answer
`206`, so step 4 of this strategy can be completed for them now and only the one
checkpoint waits. Request official access or an upstream checksum. Where neither
arrives, take route 2 or route 3 from "Where to get a checkpoint" and carry that
route's stated limit into the binding rather than into a fidelity claim.

For the already-pinned receipt, a hydrate script must have this fail-closed
shape (with every `*_SHA256` value populated, never `main` or a blank):

```bash
set -euo pipefail
root=/datavol_binder/protenix
mkdir -p "$root/checkpoint" "$root/common"

fetch_verified() {
  url=$1; destination=$2; expected=$3
  temporary="${destination}.part"
  curl --fail --location --proto '=https' --tlsv1.2 --retry 3 \
    --output "$temporary" "$url"
  printf '%s  %s\\n' "$expected" "$temporary" | sha256sum --check --status
  mv "$temporary" "$destination"
}

fetch_verified 'https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt' \
  "$root/checkpoint/protenix-v2.pt" "$PROTENIX_V2_SHA256"
# Repeat for the four direct URLs in protenix.web_service.dependency_url.URL.
```

At job time, set a no-network policy and leave `PROTENIX_ROOT_DIR` and
`TORCH_EXTENSIONS_DIR` pointing to mounted, read-only artifact and writable
kernel-cache volumes respectively. The adapter already rejects a missing target
MSA unless an operator explicitly supplies `--allow-msa-search`; the supplied
profile does not take that egress path.

## Network declaration

Separate build/hydration egress from prediction-job egress:

| Phase | Required hosts | Reason |
| --- | --- | --- |
| Fresh image build | `vemlp-cn-beijing.cr.volces.com`, `github.com`, `pypi.org`, `files.pythonhosted.org` | Upstream base image, pinned source/CUTLASS clones, and the approved hash lock. |
| Artifact hydration only | `protenix.tos-cn-beijing.volces.com` | The official checkpoint and common inference cache URLs hard-coded by the tagged source. |
| Prediction with hydrated target MSA | none | The adapter passes local A3M files; Protenix finds the checkpoint/common files locally. |

`huggingface.co`, `*.hf.co`, and `api.colabfold.com` are not part of the
Protenix v2 prediction route described here. They must not remain in a
Protenix-specific run-time egress allowlist unless some separately approved
workflow actually uses them.

## Toolcheck and eventual qualification

The profile's existing `python -m pip show protenix` is an appropriate static
toolcheck: it avoids importing CUDA-dependent code. It is not evidence that the
model loads or emits adapter-consumable output. Before promotion, run and retain
these distinct receipts in a GPU-enabled environment:

1. **Static image receipt:** base digest, source/CUTLASS commits, lock hash,
   `pip check`, `pip show protenix`, `protenix --help`, Python/Torch/CUDA
   versions, and the artifact-manifest hash.
2. **Offline load receipt:** start with egress disabled; assert CUDA is
   available; invoke a one-seed `protenix pred` after the local artifact hash
   verification; capture the first custom kernel compilation in the persistent
   extension cache. A warm repeat must not re-download the checkpoint or common
   cache and must reuse the extension cache.
3. **Adapter contract receipt:** run a minimal two-protein input containing a
   precomputed target A3M with the adapter's exact flags:

   ```text
   protenix pred --model_name protenix-v2 --seeds 0 --cycle 10 --sample 1 \
     --need_atom_confidence true --use_msa true
   ```

   Verify one `*_sample_*.cif`, one `*_full_data_sample_*.json`, and a
   `token_pair_pae` array. Then run the adapter parser and preserve the raw
   output plus its manifest. This checks the adapter's actual contract rather
   than only Protenix's own ranking score.
4. **Roster PASS decision:** record the executed command, timeouts, GPU/driver,
   image digest, exact model checksum, and output hashes. Do not set the route
   to qualified or enable a production profile solely from the static check.

The source defaults are bf16, 200 diffusion steps, five samples, and ten
cycles. The adapter deliberately specifies one sample and ten cycles; it relies
on the bf16/200-step defaults. Do not add `--use_default_params` as a casual
substitute during qualification: capture the exact adapter argv that production
will use.

## Licence evidence

At the pinned source commit, the repository's `LICENSE` is Apache License 2.0.
The upstream README states that the Protenix project, including both code and
model parameters, is released under Apache 2.0 and is free for academic and
commercial use. For redistribution, retain Apache notices and comply with its
notice/attribution requirements. This is technical evidence, not legal advice;
the actual weight receipt must still name the object bytes that were deployed.

## What you must supply before this route runs

`envs/binder_scoring_gpu.py` is fail-closed on three unset values: `_PROTENIX_WEIGHTS_SHA256`,
`_PROTENIX_REQUIREMENTS_LOCK_SHA256`, and `_PROTENIX_IMAGE_RECIPE_COMPLETE`. `build()` raises
`binder_scoring_gpu: Protenix v2 remains fail-closed` until all three carry reviewed values, and
`HYDRATE` exits with its missing-receipt message until the weights digest is recorded. Capture the
official object once through the hash-checked direct-URL route above, review the receipt, then
commit the digest. The 403 recorded at the top of this page blocks that capture today.

Then fill the `__REQUIRED__` fields on the `protenix-v2-predictor` binding: `model_revision` as
`protenix-v2.pt@sha256:<recorded value>`, `environment_identity` as the final environment spec,
image, and artifact-manifest digest, and `source_revision` as
`bytedance/Protenix@2475421477ab414b571149ad4a875c390ff8a35d` where a profile leaves it unset. Keep
`binding_status: operator-configured` until the offline GPU qualification receipts exist.

Keep the route's network declaration split: hydration reaches
`protenix.tos-cn-beijing.volces.com`, and prediction runs offline. An operator taking route 3
adds `huggingface.co` to the hydration phase only. Neither host belongs in prediction-time
egress, and ColabFold belongs in neither phase.

Do not promote a Protenix binding whose model revision lacks a SHA-256, whose source revision is
not the pinned commit, whose run-time egress includes the Hugging Face route, or that carries no
offline GPU qualification receipt.

## First-party source evidence

- [Protenix v2.0.0 release](https://github.com/bytedance/Protenix/releases/tag/v2.0.0)
  and its [peeled source commit](https://github.com/bytedance/Protenix/tree/2475421477ab414b571149ad4a875c390ff8a35d).
- Pinned [README](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/README.md),
  [setup.py](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/setup.py),
  [requirements.txt](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/requirements.txt),
  and [Dockerfile](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/Dockerfile).
- Pinned [model definitions](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/configs/configs_model_type.py),
  [inference cache downloader](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/runner/inference.py),
  [official dependency URL map](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/protenix/web_service/dependency_url.py),
  and [kernel instructions](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/docs/kernels.md).
- Pinned [LICENSE](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/LICENSE)
  and [training/inference instructions](https://raw.githubusercontent.com/bytedance/Protenix/2475421477ab414b571149ad4a875c390ff8a35d/docs/training_inference_instructions.md).
