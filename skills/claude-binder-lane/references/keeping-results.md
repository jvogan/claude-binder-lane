# Keeping results

The executor writes `keep-list.json` in the run root. It records known run files with category, path, status, and, for available files, a content hash and byte count.

## What the list holds

The list carries nine categories.

| Category | What it holds |
| --- | --- |
| `ranked-results` | `artifacts/scores/ranked-candidates.json`. |
| `scores` | Everything under `artifacts/scores/`, the output check, and each observation row's metric source file. |
| `structures` | Every `.pdb`, `.cif`, and `.mmcif` file under `artifacts/`, plus the design pose and predicted complex each observation row names. |
| `confidence` | The PAE matrix each observation row names. |
| `sequences` | Every `.fasta`, `.fa`, and `.faa` file under `artifacts/`. |
| `images` | The viewer manifest, the viewer page, and every `.png` file under `artifacts/`. |
| `receipts` | Everything under `artifacts/receipts/`. |
| `spend-ledger` | `artifacts/spend.jsonl`. |
| `run-metadata` | The run pointer, the checkpoint, `status.json`, the resolved runtime config, and the artifact index. |

A prediction stage names its predicted complexes and PAE matrices in its own observation rows rather than in a stage output manifest. The keep list reads those rows, so a structure an adapter wrote outside `artifacts/` still reaches the handoff. A path outside the run root is refused, because the list addresses every file relative to that root.

The keep list is not the artifact index. `artifacts/artifact-index.json` hashes every file in the run root. The keep list names the subset a consumer stages for handoff, so a file absent from it is deleted at worker teardown.

The `stage_flat_artifacts` function copies selected files into an empty destination directory using collision-free path-derived names [artifact_handoff.py](../claude_binder/artifact_handoff.py). Use it after an authorized run for flat artifact archiving:

```python
import json
from pathlib import Path

from claude_binder.artifact_handoff import stage_flat_artifacts

run_root = Path("<run-root>")
keep_list_path = run_root / "keep-list.json"
keep_list = json.loads(keep_list_path.read_text(encoding="utf-8"))
available = [
    Path(item["absolute_path"])
    for item in keep_list["files"]
    if item["status"] == "available"
]
staged = stage_flat_artifacts(
    available,
    source_root=run_root,
    destination=run_root / "flat-handoff",
)
```

Preserve raw receipts, configuration, and plan files alongside generated output artifacts.

The hosted workspace retains nothing. Everything under `~/.claude-science/orgs/<org_id>/workspaces/<frame_id>/` is swept when the session ends, so a file that was never saved is gone.

Saved artifacts are the durable surface. Their retention is a per-artifact declaration. `working_data` keeps only the newest version and prunes the rest on every save. `snapshot` keeps every version, and it is the only value permitted for a file the user uploaded or branched. There is no snapshot count limit; the limit that binds is storage, because every save keeps a full copy and a save that would stack past the per-project ceiling is refused with a message naming the bytes already held under that filename.

Save a run root as one directory. It is stored as a single `.tar`, which avoids repeatedly stacking large files under one name.

Source: [`artifact_handoff.py`](../claude_binder/artifact_handoff.py) defines flat handoff behavior. [`lane.py`](../claude_binder/lane.py) defines the keep-list schema.
