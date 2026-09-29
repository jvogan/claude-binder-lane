# Reading a finished run in Claude Science

A completed run leaves a self-contained report on disk. This page says what that report contains,
what it leaves out, and which shipped Claude Science skills turn it into something a reader outside
the session can use.

Nothing here interprets a result. A candidate, sequence, structure, score, rank, or picture from
this package is a computational record and does not establish binding, affinity, or biological
activity.

## Contents

- [What the run leaves you](#what-the-run-leaves-you)
- [The viewer](#the-viewer)
- [Sequences](#sequences)
- [Turning the run into a figure or a write-up](#turning-the-run-into-a-figure-or-a-write-up)
- [Querying the session itself](#querying-the-session-itself)
- [What the platform does not ship](#what-the-platform-does-not-ship)

## What the run leaves you

`render-viewer` is the last stage. After it commits, the run root holds these under `artifacts/`.

| Path | What it is |
| --- | --- |
| `viewer/index.html` | One self-contained page. The structures are inline, so it needs no server and no network. |
| `viewer/manifest.json` | One record per shown design, with rank, gates, per-seed scores, and paths. |
| `viewer/thumbnails/rank-NNN-<candidate>.png` | One ranked thumbnail per shown design. |
| `scores/ranked-candidates.json` | The ranked table the viewer and every downstream reader read. |
| `keep-list.json` | The categorized file list for handoff, including `sequences`, `structures`, and `images`. |

Each manifest design record carries `rank`, `rank_score`, `ipsae_min_ensemble`, `sc_dockq_ensemble`,
`gates`, `score_gating`, `predictor_agreement`, `per_seed_by_predictor`, `thumbnail_path`,
`design_pose_path`, `sequence_path`, `pymol_script`, and `chimerax_script`.

Read `best_design_claim_status` and `best_design_claim_reason` on the manifest before calling
anything a winner. The manifest states its own claim limit, and a rank of 1 is not by itself a
unique best design.

## The viewer

The page shows structure views and the interface metrics: ipSAE, sc_DockQ, clash count, and pLDDT.
Every shown design has a thumbnail and a viewer entry.

The manifest's `how_to_open` names an absolute path and tells you to open it in a browser. That
instruction was written for a workstation. Inside Claude Science the run root is in the frame
workspace, so prefer one of these.

- **Show the page.** Read `viewer/index.html` and surface it to the user as an artifact. It is one
  file with everything inline, so it survives being moved.
- **Show the ranked thumbnails.** `viewer/thumbnails/` is already ordered by rank, which is usually
  the fastest answer to "what did this run produce".
- **Hand over a per-design scene.** Each design record names a `pymol_script` and a
  `chimerax_script`. Those drive a local PyMOL or ChimeraX and are the route to an interactive view.
  Neither program is shipped by the platform. Both are `conditional_local_install` in
  [the catalogue](tool-catalogue.md).

Copy anything load-bearing out of the frame workspace in the same session. A workspace does not
outlive the frame, and the run record on disk goes with it.

## Sequences

The viewer page shows structures and metrics. It renders no sequences. The manifest carries a
`sequence_path` per design, so every sequence is in the run and reachable by that path.

Two ways to read them.

- **Per design.** Read the `sequence_path` each manifest record names.
- **As a ranked table.** `python -m claude_binder.lane sequence-score-table --ranked
  artifacts/scores/ranked-candidates.json --n <count>` prints a TSV whose first columns are `rank`,
  `candidate_id`, `sequence`, and `sequence_length`.

The table is complete but hard to read. It emits every score column the ranked output carries,
which runs to hundreds of columns and tens of kilobytes for a handful of rows. Select the columns
you want before showing it to anyone. `--json` gives the same rows in a form that is easier to
narrow.

Nothing in the package renders sequence and structure in one view. A reader who wants the
sequence beside the pose has to join `sequence_path` to `thumbnail_path` themselves.

## Turning the run into a figure

Two shipped skills compose here, and they are tiered. Run the composer first.

| Skill | Use it for | Tier |
| --- | --- | --- |
| `figure-composer` | Building one multi-panel figure from a one-line claim plus data references | Outer |
| `figure-style` | Correctness and legibility rules for a single final plot | Inner |

`figure-composer` takes a **claim** and **data references as artifact version IDs**, not file paths.
So promote the ranked table to a Claude Science artifact first, then hand the composer its version
ID. It loads `figure-style` itself for every panel, and it caps its own review loop at three rounds.

For one standalone plot, skip the composer and load `figure-style` alone.

## Querying the session itself

`self-awareness` documents `host.query()`, a read-only SQLite surface over Claude Science's own
metadata. Use it for run history, token usage, cost accounting, the execution log, and artifact
metadata. It answers questions like which cells ran, what was written, and when.

Two constraints it states. It is available only through the `repl` tool, not `python` or `r`. Its
results are scoped to the current project, and schema-qualified table names are rejected.

`host.query()` reports token and session accounting. It does not report provider compute cost.
Modal and fal dollars are reconciled through [approval and spend](approval-and-spend.md), and a
token meter is not a bill.

## What the platform does not ship

Claude Science ships 29 skills. None of them is a structure or sequence viewer. Structure viewing
comes from this package's own `render-viewer` output plus a local PyMOL or ChimeraX.

Four shipped model skills are outside binder work: `borzoi` and `evo2` for genomic sequence, and
`scgpt` and `scvi-tools` for single-cell expression.
