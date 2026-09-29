# Package loading

`kernel.py` defines `binder_version()` and `binder_cli()` in the Python kernel. Call `binder_version()` before starting pipeline work to verify the loaded path, package version, and load source.

The version result includes `loaded_build_manifest`, `disk_build_manifest`, and
`restart_required`. A true restart value means the skill was replaced after this
kernel loaded its package. Start a fresh conversation before the next campaign.
Reloading the sidecar alone can retain cached submodules. A null value means a
build identity is unavailable, including an environment-only installation.
Matching markers identify a build; they do not check for edits made without
updating its marker.

For one tool, call `binder_tool_info("bindcraft2")`. The result contains its
`catalog` record and `inventory` summary. Read the availability note for recorded
execution evidence. Read the live summary for routes observed in this session.
The lookup performs no provider call.

The sidecar checks for a sibling `claude_binder/` directory beside `kernel.py` and imports that directory first. When no sibling directory exists, it imports the installed package.

The kernel import does not make the package available to child processes. Set `PYTHONPATH` to the skill root before invoking stage subprocesses:

```python
import os
from pathlib import Path

skill_root = Path(binder_version()["path"]).resolve().parent.parent
os.environ["PYTHONPATH"] = os.pathsep.join(
    item for item in (str(skill_root), os.environ.get("PYTHONPATH", "")) if item
)
```

If the host rejects the sidecar, inspect the host skill-load message before calling either helper.

Use the build manifests to identify the installed distribution. A package version alone cannot distinguish builds with different files. Start a fresh conversation after replacing the skill so its guidance and runtime load together.

Source: [`kernel.py`](../kernel.py) defines the loading order and helpers. [`test_installed_skill_free_run.py`](evidence/tests/test_installed_skill_free_run.py) verifies child-process execution under `PYTHONSAFEPATH=1`.
