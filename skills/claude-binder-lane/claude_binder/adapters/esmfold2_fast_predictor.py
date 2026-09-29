#!/usr/bin/env python3
"""ESMFold2-Fast arm of the binder lane cofold stages.

This mode is the same package, image, call, and embedding lineage as
`esmfold2_predictor.py`, so it imports that module and changes two things. The
checkpoint becomes `biohub/ESMFold2-Fast`, and no chain gets an MSA.

ESMFold2-Fast and ESMFold2-Full are two modes of one predictor lineage. Their
agreement measures variation within that lineage.

The MSA difference is the one to guard. The released Fast checkpoint has no MSA
encoder, and passing an a3m is a silent no-op. An arm that reads the shared MSA
cache and hands Fast an a3m produces numbers, and the numbers are the
single-sequence numbers. So this script exposes no MSA argument at all, and
argparse rejects `--target-msa-a3m` with an error rather than accepting it and
ignoring it.

Sources: protocol lines 70 and 111, `esmfold2/SKILL.md:200`, and
`ref/docs/LOOKUP_TABLES.md`, first table, row `ef2fast`. The published row folds
both chains single sequence, target included.
"""

from . import esmfold2_predictor

# TODO The published `ef2fast` row lists `msa_max_depth=2048`, `lm_dropout=0.3`
# and `msa_column_mask_rate=0.1` among its key flags, on a model the same row
# says has no MSA encoder. That reads like a cell carried across from the
# `ef2full` row. None of the three is passed here. `esmfold2/SKILL.md` states
# that Fast does not support MSA at all, which is a third source agreeing that
# an MSA depth and a column-masking rate have nothing to act on here. It still
# does not say the settings are inert rather than rejected, so the TODO stands.
# This is TODO 3 of `report_arms.md` section 4.1.

ARM_FAST = esmfold2_predictor.ArmSpec(
    predictor_id="esmfold2-fast",
    adapter_id="esmfold2-fast-predictor",
    checkpoint="biohub/ESMFold2-Fast",
    uses_target_msa=False,
    description=(
        "ESMFold2-Fast, single sequence on every chain including the target. "
        "The checkpoint has no MSA encoder, so this arm takes no a3m."
    ),
)


def main() -> int:
    return esmfold2_predictor.main(ARM_FAST)


if __name__ == "__main__":
    raise SystemExit(main())
