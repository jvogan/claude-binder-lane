"""Runtime bridge between the adapter contract and the native design loop.

The CPU preparation path builds all aiming and sequence constraints before it
imports Torch. The GPU path then loads models and executes the sister loop.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from .constraints import ContactMask, GradientMask, build_gradient_mask, build_inter_contact_masks
from .prosite import build_prosite_prompt, overlay_signatures


_ANTIBODY_TEMPLATES: dict[str, tuple[str, Mapping[str, tuple[int, int]]]] = {
    "trastuzumab": (
        "EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}YIHWVRQAPGKGLEWVARI{hcdr2}TRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSRSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
    ),
    "atezolizumab": (
        "EVQLVESGGGLVQPGGSLRLSCAAS{hcdr1}WIHWVRQAPGKGLEWVAWI{hcdr2}TYYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (9, 15), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
    ),
    "ocankitug": (
        "QVQLVQSGAEVKKPGSSVKVSCKAS{hcdr1}WMHWVRQAPGQGLEWMGII{hcdr2}TSLNQKFQGRVTITADTSTSTAYMELSSLRSEDTAVYYCAR{hcdr3}WGQGTLVTVSSGGGSGGGSGGGSGGGSDIQMTQSPSSLSASVGDRVTITC{lcdr1}WYQQKPGKAPKLLIY{lcdr2}GVPSRFSGSGTDFTLTISSLQPEDFATYYC{lcdr3}FGQGTKVEIK",
        {"hcdr1": (7, 9), "hcdr2": (5, 6), "hcdr3": (8, 14), "lcdr1": (11, 16), "lcdr2": (7, 7), "lcdr3": (9, 9)},
    ),
}


@dataclass(frozen=True)
class RuntimePreparation:
    """CPU-safe inputs consumed by one GPU-native design seed."""

    request: Any
    seed: int
    binder_prompt: str
    position_allowed: Mapping[int, frozenset[str]]
    prosite_layout: Mapping[str, Any] | None
    gradient_mask: GradientMask
    contact_masks: ContactMask


def _antibody_prompt(framework: str, seed: int) -> str:
    try:
        template, ranges = _ANTIBODY_TEMPLATES[framework]
    except KeyError as exc:
        raise ValueError(f"unsupported antibody framework: {framework}") from exc
    random_source = random.Random(seed)
    return template.format(**{name: "#" * random_source.randint(low, high) for name, (low, high) in ranges.items()})


def binder_prompt_for_request(request: Any, seed: int) -> str:
    """Build the sister prompt family selected by the adapter request."""

    if request.binder_mode == "minibinder":
        if request.min_length is None or request.max_length is None:
            raise ValueError("minibinder request omits a length range")
        length = random.Random(seed).randint(request.min_length, request.max_length)
        return "#" * length
    if request.binder_mode == "antibody_framework" and request.antibody_framework is not None:
        return _antibody_prompt(request.antibody_framework, seed)
    raise ValueError(f"unsupported binder mode: {request.binder_mode!r}")


def prepare_runtime_request(request: Any, *, seed: int | None = None) -> RuntimePreparation:
    """Build prompt, PROSITE, gradient, and contact masks before GPU imports.

    The returned `contact_masks` is passed directly to
    `losses.compute_structure_losses` by `design.design_binder`.
    """

    selected_seed = request.seed_base if seed is None else seed
    prompt = binder_prompt_for_request(request, selected_seed)
    layout: Mapping[str, Any] | None = None
    position_allowed: Mapping[int, frozenset[str]] = {}
    if request.pattern is not None:
        layout = build_prosite_prompt(
            request.pattern,
            len(prompt),
            gap=request.pattern_gap,
            anchor=request.pattern_anchor,
            start=request.pattern_start,
        )
        overlay = overlay_signatures(prompt, [layout])
        prompt = str(overlay["prompt"])
        position_allowed = dict(overlay["position_allowed"])
    gradient_mask = build_gradient_mask(
        prompt,
        allowed_amino_acids=request.allowed_amino_acids,
        position_allowed=position_allowed,
    )
    if not gradient_mask.mutable_positions:
        raise ValueError("the native design prompt has no mutable positions")
    contact_masks = build_inter_contact_masks(
        len(request.target_sequence.replace("|", "")),
        len(prompt),
        request.epitope_indices_0based,
    )
    return RuntimePreparation(
        request=request,
        seed=selected_seed,
        binder_prompt=prompt,
        position_allowed=position_allowed,
        prosite_layout=layout,
        gradient_mask=gradient_mask,
        contact_masks=contact_masks,
    )


def _run_torch_design(preparation: RuntimePreparation, config: Any) -> Mapping[str, Any]:
    """Load models and execute one prepared seed after CPU validation succeeds."""

    from .design import design_binder
    from .models import ESMFold2Designer

    request = preparation.request
    designer = ESMFold2Designer(config)
    designer.load(request.use_scaling_critics)
    sequences, trajectory, critic_results = design_binder(
        designer.inversion_models,
        designer.hf_critic_models,
        designer.esmc_model,
        target_sequence=request.target_sequence,
        binder_sequence=preparation.binder_prompt,
        is_antibody=request.binder_mode == "antibody_framework",
        seed=preparation.seed,
        batch_size=request.batch_size,
        allowed_amino_acids=request.allowed_amino_acids,
        position_allowed=preparation.position_allowed,
        contact_masks=preparation.contact_masks,
        steps=config.steps,
        learning_rate=config.learning_rate,
        temperature_min=config.temperature_min,
        plm_weight=config.plm_weight,
        plm_weight_antibody=config.plm_weight_antibody,
        esmc_mask_fraction=config.esmc_mask_fraction,
        critic_num_loops=config.critic_num_loops,
        critic_num_sampling_steps=config.critic_num_sampling_steps,
        loss_weights={
            "intra_contact": config.intra_contact_weight,
            "inter_contact": config.inter_contact_weight,
            "glob": config.globularity_weight,
        },
        save_confidence_arrays=config.save_confidence_arrays,
    )
    candidates: list[dict[str, Any]] = []
    complexes: dict[str, str] = {}
    logits: dict[str, Any] = {}
    by_batch: dict[int, list[Mapping[str, Any]]] = {}
    for record in critic_results:
        by_batch.setdefault(int(record["batch_idx"]), []).append(record)
    for batch_index, sequence in enumerate(sequences):
        records = by_batch.get(batch_index, [])
        if not records:
            raise ValueError(f"native critic pass returned no result for batch index {batch_index}")
        primary = records[0]
        candidate_id = f"native-{preparation.seed:06d}-{batch_index:03d}"
        complexes[candidate_id] = str(primary["complex"])
        logits[candidate_id] = primary["logits"]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "binder_sequence": sequence.split("|")[-1],
                "complex_sequence": sequence,
                "generator_seed": preparation.seed,
                "batch_index": batch_index,
                "aiming_status": request.aiming_status,
                "epitope_indices_0based": list(preparation.contact_masks.epitope_indices_0based),
                "critic_scores": [
                    {
                        "critic_name": record["critic_name"],
                        "iptm": record.get("iptm"),
                        "distogram_iptm_proxy": record.get("distogram_iptm_proxy"),
                        "cdr_distogram_iptm_proxy": record.get("cdr_distogram_iptm_proxy"),
                        "final_loss": record.get("final_loss"),
                    }
                    for record in records
                ],
            }
        )
    return {"candidates": candidates, "complexes": complexes, "trajectory": {str(preparation.seed): trajectory}, "logits": logits}


def run_native_design(*, request: Any, config: Any, kernel_backend: None = None) -> Mapping[str, Any]:
    """Run all requested native-design seeds and return adapter artifacts.

    The adapter always passes `kernel_backend=None` because Experimental models
    require the reference kernel path.
    """

    if kernel_backend is not None:
        raise ValueError("ESMFold2 Experimental design variants require kernel_backend=None")
    if config.steps is None or config.critic_num_sampling_steps is None:
        raise ValueError("native design requires campaign values for steps and critic_num_sampling_steps")
    candidates: list[Mapping[str, Any]] = []
    complexes: dict[str, str] = {}
    logits: dict[str, Any] = {}
    trajectories: dict[str, Any] = {}
    for seed_offset in range(request.seeds):
        preparation = prepare_runtime_request(request, seed=request.seed_base + seed_offset)
        result = _run_torch_design(preparation, config)
        candidates.extend(result["candidates"])
        complexes.update(result["complexes"])
        logits.update(result["logits"])
        trajectories.update(result["trajectory"])
    if len(candidates) != request.n_designs:
        raise ValueError(f"native design returned {len(candidates)} candidates for requested {request.n_designs}")
    return {"candidates": candidates, "complexes": complexes, "trajectory": trajectories, "logits": logits}
