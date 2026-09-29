"""Locate files that sit beside the package in one layout and above it in another.

`run_intent.py` lives at `skills/claude-binder-lane/run_intent.py` in this repository
and at the skill root in an installed skill, because the build flattens the two. Tests
that hard-coded the repository shape resolved to a directory that does not exist once
installed, and the installed copy reported twenty-six collection errors for that reason
alone. A user's agent running the skill's own self-test would read that as a broken
skill.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SKILL_SUBDIRECTORY = ("skills", "claude-binder-lane")
#: The file that marks a skill root in both layouts. `run_intent.py` sits at the
#: skill root when installed and under `skills/claude-binder-lane/` in the
#: repository, and no other directory in either tree holds one.
SKILL_MARKER = "run_intent.py"


def _roots() -> tuple[Path, Path]:
    """Find the installed and repository skill roots by walking up from this file.

    Fixed depths cannot do this. The build ships this helper to
    `references/evidence/tests/`, three levels below the skill root, while the
    repository keeps it at `references/evidence/tests/`, three levels below the
    repository root. A `parents[1]` that is right in one tree is wrong in the
    other, and the two shipped tests that import this module failed to collect
    in an installed skill for that reason.
    """
    installed = repository = None
    for ancestor in Path(__file__).resolve().parents:
        if installed is None and (ancestor / SKILL_MARKER).is_file():
            installed = ancestor
        if repository is None and (
            ancestor.joinpath(*SKILL_SUBDIRECTORY, SKILL_MARKER)
        ).is_file():
            repository = ancestor
        if installed is not None and repository is not None:
            break
    # Keep the historical values as the fallback so a tree that holds neither
    # marker reports the same paths in its error as it always did.
    return (
        installed if installed is not None else PACKAGE_ROOT.parent,
        repository if repository is not None else PACKAGE_ROOT.parents[1],
    )


INSTALLED_ROOT, REPOSITORY_ROOT = _roots()


def skill_file(name: str) -> Path:
    """Return the skill file called ``name``, whichever layout this copy is in.

    The installed layout is checked first, because that is what a user runs.
    """
    candidates = (
        INSTALLED_ROOT / name,
        REPOSITORY_ROOT.joinpath(*SKILL_SUBDIRECTORY, name),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"skill file {name} is in neither layout; looked in {searched}")


def skill_root() -> Path:
    """Return the directory that holds the skill's own files, in either layout."""
    installed = INSTALLED_ROOT
    repository = REPOSITORY_ROOT.joinpath(*SKILL_SUBDIRECTORY)
    if (installed / "run_intent.py").is_file():
        return installed
    if (repository / "run_intent.py").is_file():
        return repository
    raise FileNotFoundError(
        f"no skill root in either layout; looked in {installed} and {repository}"
    )
