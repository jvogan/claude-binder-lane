#!/usr/bin/env python3
"""Call one endpoint of a user-selected ESMFold2-Full app and save the response.

The client reads a named variable from its own environment, defaulting to
``FAL_KEY``. Where the caller already carries it, run the client directly. Where
it does not, run it through the API
lane, which supplies the credential without putting it in a command line, a
shell history, or a process listing:

    fal-credential-wrapper exec-model -- python3 call.py hydrate --out-dir runs/hydrate-01
    fal-credential-wrapper exec-model -- python3 call.py predict \
        --binder-fasta binder.fasta --target-fasta target.fasta \
        --out-dir runs/predict-01

The request pattern matches the qualified fal client in the vendor tree: the
URL is checked before it is used, redirects are refused, and the no-retry and
disable-fallback headers keep the receipt unambiguous.
"""

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

FAL_HOST = "fal.run"
DEFAULT_CREDENTIAL_ENV = "FAL_KEY"
ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "fal redirect rejected", headers, fp
        )


def validate_fal_url(fal_url):
    parsed = urllib.parse.urlparse(fal_url)
    try:
        port = parsed.port
    except ValueError:
        port = -1
    segments = [segment for segment in parsed.path.split("/") if segment]
    if (
        parsed.scheme != "https"
        or parsed.hostname != FAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.path != "/" + "/".join(segments)
        or len(segments) != 2
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(
            "--fal-url must be exactly https://" + FAL_HOST + "/<team>/<app>"
            + " with no credentials, query, or fragment"
        )
    return "https://" + FAL_HOST + "/" + "/".join(segments)


def endpoint_url(fal_url, endpoint):
    return validate_fal_url(fal_url) + "/" + endpoint


def credential_environment_name(value):
    """Validate an environment-variable name without reading its value."""
    if not isinstance(value, str) or ENVIRONMENT_NAME_RE.fullmatch(value) is None:
        raise SystemExit(
            "--credential-env must be an environment-variable name such as FAL_KEY"
        )
    return value


def call(
    url,
    payload,
    timeout_seconds,
    out_dir,
    credential_env=DEFAULT_CREDENTIAL_ENV,
):
    credential_env = credential_environment_name(credential_env)
    fal_key = os.environ.get(credential_env)
    if not fal_key:
        raise SystemExit(
            credential_env + " is unavailable. Set it in this process's environment, or run "
            "this client through the credential wrapper: fal-credential-wrapper exec-model -- ... "
            "Credential presence proves only presence. The provider authorization "
            "preflight verifies application access."
        )
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": "Key " + fal_key,
            "Content-Type": "application/json",
            "X-Fal-No-Retry": "1",
            "X-Fal-Request-Timeout": str(timeout_seconds),
            "X-App-Fal-Disable-Fallback": "1",
        },
    )
    opener = urllib.request.build_opener(RejectRedirects())
    try:
        with opener.open(request, timeout=timeout_seconds + 60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        # The endpoints now answer a failure with a dict detail carrying the
        # stage, the log and a traceback tail. Writing it out before exiting
        # is the difference between a diagnosis and another blind cold start.
        detail = error.read().decode("utf-8", errors="replace")
        (out_dir / "error.txt").write_text(
            "HTTP " + str(error.code) + "\n" + detail + "\n"
        )
        try:
            parsed = json.loads(detail)
        except ValueError:
            parsed = None
        if parsed is not None:
            (out_dir / "error.json").write_text(json.dumps(parsed, indent=2) + "\n")
            inner = parsed.get("detail")
            if isinstance(inner, dict):
                print(json.dumps(inner, indent=2, sort_keys=True))
                raise SystemExit(
                    "fal request failed with HTTP " + str(error.code)
                    + " in stage " + str(inner.get("stage"))
                    + ": " + str(inner.get("error"))
                )
        raise SystemExit(
            "fal request failed with HTTP " + str(error.code) + ": " + detail[-8000:]
        ) from error


def read_sequence(path):
    """Read one sequence from a FASTA file or a plain text file."""
    text = Path(path).read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) > 1:
        raise SystemExit(str(path) + " holds more than one FASTA record")
    sequence = "".join(line for line in lines if not line.startswith(">"))
    if not sequence.isalpha() or not sequence.isupper():
        raise SystemExit(str(path) + " is not a single uppercase sequence")
    return sequence


def read_msa(path):
    """Return the uncompressed A3M bytes and an integrity digest."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            payload = handle.read()
    else:
        payload = path.read_bytes()
    if not payload:
        raise SystemExit(str(path) + " is empty")
    return payload, hashlib.sha256(payload).hexdigest()


def save_blob(record, stem, out_dir):
    """Write a compressed array blob out as raw bytes plus a sidecar."""
    raw = zlib.decompress(base64.b64decode(record["data"]))
    if len(raw) != record["raw_bytes"]:
        raise SystemExit(stem + " blob decompressed to the wrong length")
    if hashlib.sha256(raw).hexdigest() != record["raw_sha256"]:
        raise SystemExit(stem + " blob failed its SHA-256 check")
    (out_dir / (stem + ".bin")).write_bytes(raw)
    (out_dir / (stem + ".json")).write_text(
        json.dumps(
            {
                "dtype": record["dtype"],
                "shape": record["shape"],
                "byte_order": "little",
                "note": "read with numpy.fromfile(path, dtype).reshape(shape)",
            },
            indent=2,
        )
        + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", choices=["preflight", "hydrate", "predict"])
    parser.add_argument("--fal-url", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--credential-env",
        type=credential_environment_name,
        default=DEFAULT_CREDENTIAL_ENV,
        help=(
            "Environment-variable name holding the fal credential. The value is read "
            "only inside this process and is never accepted on the command line."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1200)
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
                "predict needs --binder-fasta, --target-fasta and --target-msa-a3m"
            )
        msa_bytes, msa_sha256 = read_msa(args.target_msa_a3m)
        payload = {
            "binder_seq": read_sequence(args.binder_fasta),
            "target_seq": read_sequence(args.target_fasta),
            "target_msa_a3m_b64": base64.b64encode(msa_bytes).decode("ascii"),
            "target_msa_sha256": msa_sha256,
            "target_msa_bytes": len(msa_bytes),
            "seed": args.seed,
            "max_seconds": args.max_seconds,
        }

    url = endpoint_url(args.fal_url, args.endpoint)
    result = call(
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
            "venv_complete": result.get("venv_complete"),
            "total_bytes_on_disk": result.get("total_bytes_on_disk"),
            "repos": result.get("repos"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "log": result.get("log"),
        }
    else:
        (args.out_dir / "predicted.cif").write_text(result["mmcif"])
        save_blob(result["pae"], "pae", args.out_dir)
        save_blob(result["plddt"], "plddt", args.out_dir)
        summary = {
            "best_index": result["best_index"],
            "iptm": result["iptm"],
            "ptm": result["ptm"],
            "mean_plddt": result["mean_plddt"],
            "chains": result["chains"],
            "expected_chain_lengths": result["expected_chain_lengths"],
            "samples": result["samples"],
            "msa": result["msa"],
            "nonfinite_svd_repairs": result["nonfinite_svd_repairs"],
            "environment": result["environment"],
            "resolved_snapshots": result["resolved_snapshots"],
            "timings": result["timings"],
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
