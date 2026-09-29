#!/usr/bin/env python3
"""Call a user-selected AlphaFold2-Multimer-v3 fal app and save one response.

The deployed app receives the target sequence, the designed binder sequence, and
the target-chain A3M used for the fold. It returns one mmCIF structure plus
compressed PAE and pLDDT arrays. The client keeps credentials in its inherited
environment and never writes the credential into an argv or an output file.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from claude_binder.clients import fal_esmfold2_full_client as fal_common


MODEL_TYPE = "alphafold2_multimer_v3"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", choices=["preflight", "hydrate", "predict"])
    parser.add_argument("--fal-url", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--credential-env",
        type=fal_common.credential_environment_name,
        default=fal_common.DEFAULT_CREDENTIAL_ENV,
        help=(
            "Environment-variable name holding the fal credential. The value is read "
            "only inside this process and is never accepted on the command line."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--budget-seconds", type=int, default=1000)
    parser.add_argument("--rebuild-environment", action="store_true")
    parser.add_argument("--raise-test", action="store_true")
    parser.add_argument("--binder-fasta", type=Path)
    parser.add_argument("--target-fasta", type=Path)
    parser.add_argument("--target-msa-a3m", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.endpoint == "preflight":
        payload = {"raise_test": args.raise_test}
    elif args.endpoint == "hydrate":
        payload = {
            "budget_seconds": args.budget_seconds,
            "rebuild_environment": args.rebuild_environment,
        }
    else:
        if not args.binder_fasta or not args.target_fasta or not args.target_msa_a3m:
            raise SystemExit(
                "predict needs --binder-fasta, --target-fasta, and --target-msa-a3m"
            )
        msa_bytes, msa_sha256 = fal_common.read_msa(args.target_msa_a3m)
        payload = {
            "model_type": MODEL_TYPE,
            "binder_seq": fal_common.read_sequence(args.binder_fasta),
            "target_seq": fal_common.read_sequence(args.target_fasta),
            "target_msa_a3m_b64": base64.b64encode(msa_bytes).decode("ascii"),
            "target_msa_sha256": msa_sha256,
            "target_msa_bytes": len(msa_bytes),
            "seed": args.seed,
            "max_seconds": args.max_seconds,
        }

    url = fal_common.endpoint_url(args.fal_url, args.endpoint)
    result = fal_common.call(
        url,
        payload,
        args.timeout_seconds,
        args.out_dir,
        credential_env=args.credential_env,
    )
    (args.out_dir / "response.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.endpoint == "preflight":
        summary = result
    elif args.endpoint == "hydrate":
        summary = {
            "complete": result.get("complete"),
            "stage": result.get("stage"),
            "next_action": result.get("next_action"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        }
    else:
        (args.out_dir / "predicted.cif").write_text(result["mmcif"])
        fal_common.save_blob(result["pae"], "pae", args.out_dir)
        fal_common.save_blob(result["plddt"], "plddt", args.out_dir)
        summary = {
            "model_type": MODEL_TYPE,
            "iptm": result.get("iptm"),
            "ptm": result.get("ptm"),
            "mean_plddt": result.get("mean_plddt"),
            "chains": result.get("chains"),
            "expected_chain_lengths": result.get("expected_chain_lengths"),
            "msa": result.get("msa"),
            "environment": result.get("environment"),
            "resolved_snapshots": result.get("resolved_snapshots"),
            "timings": result.get("timings"),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
