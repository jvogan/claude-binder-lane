"""Lazy sweep expansion and candidate selection helpers from the sister project."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence


TOP_N = 84
ISOELECTRIC_POINT_MAX = 6.0
SCALING_CHECKPOINT_SUBSTRING = "ESMFold2-Experimental-Fast-base"


def expand_sweep(line_sweeps: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expand each supplied axis into one native design configuration."""

    keys = list(line_sweeps)
    return [dict(zip(keys, values)) for values in product(*(line_sweeps[key] for key in keys))]


def select_designs(
    configs: Sequence[Mapping[str, Any]],
    raw_results: Sequence[Any],
    top_n: int = TOP_N,
    isoelectric_point_max: float = ISOELECTRIC_POINT_MAX,
) -> Any:
    """Select native-design results with the sister project selection rules.

    Pandas, Biopython, and tqdm load only when a caller explicitly selects a
    sweep result. The adapter runtime does not call this optional helper.
    """

    import pandas
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    frames = [pandas.DataFrame(result[2]).assign(**config) for config, result in zip(configs, raw_results)]
    if not frames:
        return pandas.DataFrame()
    frame = pandas.concat(frames, ignore_index=True)
    frame["binder_sequence"] = frame.designed_sequence.str.split(r"\|").str[-1]
    frame["isoelectric_point"] = [ProteinAnalysis(sequence).isoelectric_point() for sequence in frame.binder_sequence]
    frame = frame[frame.is_antibody | frame.isoelectric_point.lt(isoelectric_point_max)]

    def rank_group(group: Any) -> Any:
        scaling = group.critic_name.str.contains(SCALING_CHECKPOINT_SUBSTRING, regex=False, na=False)
        proxy = group.distogram_iptm_proxy.where(~group.is_antibody, group.cdr_distogram_iptm_proxy)
        group = group.assign(iptm_score=group.iptm.where(~scaling), iptm_proxy_score=proxy.where(scaling))
        scores = group.groupby("designed_sequence", as_index=False).agg(
            iptm_score=("iptm_score", "mean"),
            iptm_proxy_score=("iptm_proxy_score", "mean"),
        )
        # The two terms are mutually exclusive by construction: iptm_score is
        # defined only for non-scaling-checkpoint rows and iptm_proxy_score only
        # for scaling rows. Filling the absent one with zero and halving both
        # divided a sequence measured by one critic family by two, so selection
        # followed which critics happened to run. Average what is present.
        scores["selection_score"] = scores[["iptm_score", "iptm_proxy_score"]].mean(axis=1)
        return scores.nlargest(min(len(scores), top_n), "selection_score")

    return (
        frame.groupby(["target_name", "binder_name"], group_keys=True)
        .apply(rank_group)
        .reset_index(level=["target_name", "binder_name"])
        .reset_index(drop=True)
    )
