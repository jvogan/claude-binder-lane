# Resume

Use `--resume` after an execution writes `partial-summary.json` in the run root. Review the reported failure, correct the input or configuration, and retain the original bundle before resuming.

```python
from pathlib import Path

bundle = Path("<run-bundle>")
run_root = Path("<run-root>")
if binder_cli([
    "execute", "--plan", str(bundle / "run-plan.json"),
    "--run-root", str(run_root), "--stage", "all", "--resume", "--json",
]) != 0:
    raise RuntimeError("resume refused or failed; read partial-summary.json")
```

The executor verifies the materialized bundle before reuse [lane.py](../claude_binder/lane.py). A modified bundle causes the resume command to refuse execution and requires a fresh materialization.

Source: [`lane.py`](../claude_binder/lane.py) defines the `--resume` argument and bundle checkpoint verification logic.
