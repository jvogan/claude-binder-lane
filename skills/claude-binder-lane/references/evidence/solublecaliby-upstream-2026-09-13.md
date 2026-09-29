# SolubleCaliby upstream evidence

**Read 2026-09-13.** This note records source and model-card inspection for the
SolubleCaliby adapter. It is not a runtime receipt. No Caliby checkpoint was
downloaded, no Caliby process was started, and no hardware or cost result is
claimed here.

## Source and code licence

- [Caliby source at `41d31560c3c73d7980d94f40f3c852b90bfab5c0`](https://github.com/ProteinDesignLab/caliby/tree/41d31560c3c73d7980d94f40f3c852b90bfab5c0)
  contains an Apache-2.0 `LICENSE`. Its
  [`pyproject.toml`](https://github.com/ProteinDesignLab/caliby/blob/41d31560c3c73d7980d94f40f3c852b90bfab5c0/pyproject.toml)
  also declares `Apache-2.0` and Python `>=3.12`.
- This supports the catalog's code licence and the adapter's separate
  `--tool-python` requirement. It does not establish an installed environment.

## Weights and checkpoint

- The [weights repository at `a51f011f4ec7ffe2daad3ab9b2bfb67ff628a096`](https://huggingface.co/ProteinDesignLab/caliby-weights/tree/a51f011f4ec7ffe2daad3ab9b2bfb67ff628a096)
  is labelled `apache-2.0`. The model API was read at
  `a51f011f4ec7ffe2daad3ab9b2bfb67ff628a096` on 2026-09-12 and reported that
  same card licence.
- At the source pin,
  [`caliby/weights.py`](https://github.com/ProteinDesignLab/caliby/blob/41d31560c3c73d7980d94f40f3c852b90bfab5c0/caliby/weights.py)
  maps `soluble_caliby_v1` to `caliby/soluble_caliby_v1.ckpt`. It downloads a
  registry name when the local file is absent, while a value ending in `.ckpt`
  is treated as a literal path.
- The adapter therefore accepts only a local checkpoint path and a caller-supplied
  SHA-256. The source and weights revisions identify the intended upstream trees;
  this repository records no checkpoint-byte digest because it has not received
  those bytes.

## Fixed-backbone and ensemble routes

- The [README at the source pin](https://github.com/ProteinDesignLab/caliby/blob/41d31560c3c73d7980d94f40f3c852b90bfab5c0/README.md)
  lists `soluble_caliby_v1` as trained on monomers and interfaces. It documents
  fixed-backbone sequence design and a separate Protpardelle-1c ensemble path.
- The README describes a primary conformer plus optional additional `.pdb` or
  `.cif` conformers, with the primary included and a maximum count of 32 by
  default. It permits a primary-only directory.
- `rfdiffusion3-two-arm.template.json` selects the adapter's fixed-backbone
  `run` subcommand. It does not select `run-ensemble` or `generate-ensembles`.
  The latter subcommands remain available for an operator who separately
  qualifies their additional dependencies and weights.

## Input and output contract

- The README says a position-constraint CSV can hold `fixed_pos_seq`, that an
  absent CSV redesigns all positions, and that numerical residue positions use
  `label_seq_id` rather than `auth_seq_id`.
- [`seq_des_utils.py`](https://github.com/ProteinDesignLab/caliby/blob/41d31560c3c73d7980d94f40f3c852b90bfab5c0/caliby/eval/eval_utils/seq_des_utils.py)
  accepts a bare chain constraint and masks every token in that chain. The
  adapter fixes whole non-design chains with that form, so it does not translate
  author residue numbers into numerical Caliby positions.
- [`seq_des.py`](https://github.com/ProteinDesignLab/caliby/blob/41d31560c3c73d7980d94f40f3c852b90bfab5c0/caliby/eval/sampling/seq_des.py)
  writes `seq_des_outputs.csv` after selecting CUDA when available and CPU
  otherwise. Its helper writes one native `.cif` per sample and records the
  path and chain-separated sequence in that CSV; the README states that output
  chains use alphabetical chain order and `:` separators.

## Remaining qualification limits

- The exact backbone atom subset accepted by the fixed-backbone entry point is
  not declared here. Its upstream mask and resolved-residue logic need an input
  probe or a fuller source reading before the catalog can state an atom set.
- A local run must still establish whether AtomWorks accepts the prepared poses,
  the observed runtime device, peak memory, checkpoint digest, output contents,
  and any measured cost.
