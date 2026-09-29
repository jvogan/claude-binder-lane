#!/usr/bin/env python3
"""Stock ESMFold2-Fast worker for small_campaign.py's explicit platform arm.

The caller supplies the model revision and captures this process's receipt.
Only prediction is done here; score orientation and chain mapping are supplied
separately to the artifact scorer after output inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        parser.error("--revision must be the pinned 40-hex ESMFold2-Fast model revision")

    import numpy as np
    import torch
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    if not torch.cuda.is_available():
        raise RuntimeError("ESMFold2-Fast platform worker needs CUDA")
    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("input must be a nonempty list")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast", revision=args.revision).cuda().eval()
    model.set_kernel_backend("fused")
    model.set_chunk_size(None)
    rows = []
    for job in jobs:
        seqs = job["sequences"]
        if len(seqs) != 2 or [s["id"] for s in seqs] != ["A", "B"]:
            raise ValueError(f"{job.get('id')}: expected target A and binder B")
        spi = StructurePredictionInput(sequences=[
            ProteinInput(id="A", sequence=seqs[0]["sequence"]),
            ProteinInput(id="B", sequence=seqs[1]["sequence"]),
        ])
        for seed in job["seeds"]:
            t0 = time.monotonic()
            result = ESMFold2InputBuilder().fold(
                model, spi, num_loops=10, num_sampling_steps=68,
                num_diffusion_samples=1, seed=int(seed),
            )
            prediction = result[0] if isinstance(result, (list, tuple)) else result
            tag = f"{job['id']}__s{seed}"
            cif = out / f"{tag}.cif"
            pae_file = out / f"{tag}.npz"
            cif.write_text(prediction.complex.to_mmcif(), encoding="utf-8")
            pae = prediction.pae.detach().cpu().float().numpy() if hasattr(prediction.pae, "detach") else prediction.pae
            np.savez_compressed(pae_file, pae=np.asarray(pae, dtype=np.float32))
            rows.append({"candidate_id": job["id"], "seed": seed, "complex_path": str(cif),
                         "pae_npz_path": str(pae_file), "seconds": round(time.monotonic()-t0, 3)})
            (out / "prediction-index.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": len(rows), "model_revision": args.revision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
