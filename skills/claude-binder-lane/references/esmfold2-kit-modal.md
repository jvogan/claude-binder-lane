# ESMFold2 optimization kit on Modal

Use the shipped [`esmfold2_kit_gpu.py`](../envs/esmfold2_kit_gpu.py) to build
Anthropic's [ESMFold2 optimization kit](https://github.com/anthropics/uplifting-biomolecular-modeling/tree/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2)
at commit `f4f62fa6592ae4938d49b1757bea0cfeff9f468e`. This is the H100/H200
`img_ef2_fa` stack. Claude Science can use it for a kit prediction and record
the result beside other predictor arms. The campaign adapter does not select
an accelerated mode automatically.

## Build once, then use the environment

The kit's [source builder](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/f4f62fa6592ae4938d49b1757bea0cfeff9f468e/esmfold2/environment/build_wheels.sh)
checks two Git commits and the flash-attn source archive digest. It compiles
xformers, TransformerEngine, and flash-attn for compute capability 9.0. A single
Modal `Image.run_commands` layer gives this expensive compile no CPU or memory
setting. The recipe therefore builds each wheel in its own sequential CPU
Sandbox and saves the wheel plus its SHA-256 digest in a persistent Volume.
The runtime image verifies those three digests, installs the wheels, runs
`pip check` and the kit's `run.sh install`, and contains no CUDA compiler.
[Modal Sandboxes](https://modal.com/docs/guide/sandboxes) take explicit CPU and
memory requests. [Modal Volumes](https://modal.com/docs/guide/volumes) commit on
Sandbox shutdown. [Image.run_function](https://modal.com/docs/guide/images)
mounts the wheel Volume during final image assembly.

The planning resource settings are 8 CPU cores and 64 GiB for each of the
first two wheels, and 16 cores and 128 GiB for flash-attn. These are recipe
settings, not measured usage. Claude Science should check current account
limits and rates, propose a spend ceiling, and adjust `STAGE_RESOURCES` or pass
an explicit `resources` mapping if needed. Compiles are sequential so the
Volume has one writer at a time. A rerun can reuse a completed wheel Volume;
the final image rehashes its contents before installation. When rebuilding
wheels, run all three stages to refresh the set together.

In a Python context that can import `modal`, load the installed recipe by its
path and run the following after the scientist approves the provider and spend
ceiling:

```python
import importlib.util

path = "/path/to/installed/claude-binder-lane/envs/esmfold2_kit_gpu.py"
spec = importlib.util.spec_from_file_location("esmfold2_kit_gpu", path)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)
kit.prepare_wheels()  # billed: three sequential CPU Sandboxes
```

`prepare_wheels()` leaves the finished files in the named
`claude-science-esmfold2-kit-wheels-h100` Volume. It terminates each Sandbox
even when a build fails. Inspect the build logs and wheel digests, then build
the runtime environment through the Claude Science Modal provider:

```python
result = build_env(
    "esmfold2_kit_gpu",
    path="/path/to/installed/claude-binder-lane/envs/esmfold2_kit_gpu.py",
    hydrate=True,
)
print(result)
```

`build_env` consumes the recipe's `build()` declaration. `hydrate=True` runs
`HYDRATE`, which calls `run.sh install --weights /weights`, retries transient
download failures up to three times, and checks the pinned files against
`stock/PINS.json` through the kit's own installer. The three pinned Biohub
snapshots total about 27 GB. Keep the returned environment identity and the
`/weights` Volume for later jobs. The first hydration may have substantial
download and image assembly time, so include it in the plan. Reuse a matching
ledger entry when the pinned image and Volume already exist.

## Check and predict

Run `CHECK` on H100 with the hydrated `/weights` Volume mounted. It resolves
the selected mode against the kit's GPU table and prints an
`[esmfold2-opt] DRY-RUN` line. Require exit code 0 and the requested variant
and mode in that line. Then submit a small input through the kit's `pred`
command on the same image and Volume:

```sh
cd /kit/esmfold2
bash run.sh pred --config h100 --variant fast --mode fast \
  --input /path/to/prediction-input.json --out_dir /path/to/output --seeds 0
```

Record the input, output, pins, variant, mode, card, seed, image identity,
elapsed time, and settled provider cost. Require exit code 0, parsed output,
and an `[esmfold2-opt] ACTIVE mode=fast variant=fast` announcement before
scaling a campaign. The kit's `NOT ACTIVE` message means that mode was refused.
The kit also has `full_msa` and `full_nomsa` variants and `off`, `exact`,
`fast`, and `big` modes. Select a mode that fits the scientific question and
the available hardware; the recipe's `check_command()` validates these names.

This image targets H100/H200. For A100, use the upstream
`img_esmfold2_a100` build stack and a matching `--config a100`; changing just
the run flag is insufficient. Keep the upstream kit's `LICENSE`, `NOTICE`,
and bundled third-party notices with a redistributed derivative.
