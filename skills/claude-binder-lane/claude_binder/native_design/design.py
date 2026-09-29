"""Gradient-guided ESMFold2 binder design with lazy tensor imports."""

from __future__ import annotations

import math
import random
from typing import Any, Iterable, Mapping

from .constants import AA_DIMS, CYS_IDX, MUTABLE_TOKEN, PROTEIN_1TO3, TOKEN_IDS, TOKENS
from .folding import complex_to_pdb_string, fold_and_get_distogram
from .losses import compute_distogram_iptm_proxy, compute_esmc_pseudoperplexity_nll, compute_structure_losses


def _imports() -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    import torch.optim as optim
    from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context

    return {"torch": torch, "functional": functional, "optim": optim, "seed_context": _seed_context}


def _aa_logit_index(residue: str) -> int:
    return TOKEN_IDS[PROTEIN_1TO3[residue]] - 2


def _disallowed_mutable_indices(allowed_amino_acids: str | None) -> list[int]:
    if allowed_amino_acids is None:
        return [CYS_IDX]
    allowed = "".join(dict.fromkeys(allowed_amino_acids.upper()))
    invalid = sorted(set(allowed) - set(PROTEIN_1TO3))
    if invalid:
        raise ValueError("allowed_amino_acids contains invalid amino acids: " + "".join(invalid))
    if not allowed:
        raise ValueError("allowed_amino_acids must not be empty")
    allowed_indices = {_aa_logit_index(residue) for residue in allowed}
    return [index for index in range(AA_DIMS) if index not in allowed_indices]


def _allowed_to_disallowed_indices(allowed: Iterable[str]) -> list[int]:
    indices = {_aa_logit_index(residue.upper()) for residue in allowed}
    if not indices:
        raise ValueError("per-position allowed set must not be empty")
    return [index for index in range(AA_DIMS) if index not in indices]


def sequence_to_one_hot(sequence: str, device: str = "cuda") -> Any:
    """Convert an amino-acid sequence into the ESMFold2 token one-hot format."""

    dependencies = _imports()
    torch = dependencies["torch"]
    functional = dependencies["functional"]
    token_indices = [TOKEN_IDS[PROTEIN_1TO3[residue]] for residue in sequence]
    return functional.one_hot(torch.tensor(token_indices), num_classes=len(TOKENS)).to(device).unsqueeze(0).float()


def build_initial_soft_sequence_logits(
    sequence: str,
    batch_size: int,
    allowed_amino_acids: str | None = None,
    position_allowed: Mapping[int, Iterable[str]] | None = None,
) -> Any:
    """Build sister-compatible logits for fixed and mutable binder positions."""

    dependencies = _imports()
    torch = dependencies["torch"]
    disallowed = _disallowed_mutable_indices(allowed_amino_acids)
    if position_allowed is None and all(residue == MUTABLE_TOKEN for residue in sequence):
        logits = 0.01 * torch.randn([batch_size, len(sequence), AA_DIMS])
        logits[:, :, disallowed] = -1e6
    else:
        logits = torch.zeros([batch_size, len(sequence), AA_DIMS])
        for index, residue in enumerate(sequence):
            if residue == MUTABLE_TOKEN:
                logits[:, index, :] = 0.01 * torch.randn(batch_size, AA_DIMS)
                blocked = _allowed_to_disallowed_indices(position_allowed[index]) if position_allowed and index in position_allowed else disallowed
                logits[:, index, blocked] = -1e6
            else:
                if residue not in PROTEIN_1TO3:
                    raise ValueError(f"binder prompt contains unsupported residue: {residue}")
                logits[:, index, _aa_logit_index(residue)] = 10.0
    return logits.requires_grad_(True)


def build_gradient_mask(
    sequence: str,
    batch_size: int,
    allowed_amino_acids: str | None = None,
    position_allowed: Mapping[int, Iterable[str]] | None = None,
) -> Any:
    """Build a `[batch, position, amino-acid]` gradient mask."""

    dependencies = _imports()
    torch = dependencies["torch"]
    blocked = _disallowed_mutable_indices(allowed_amino_acids)
    mask = torch.ones([batch_size, len(sequence), AA_DIMS])
    fixed = [index for index, residue in enumerate(sequence) if residue != MUTABLE_TOKEN]
    mask[:, fixed, :] = 0.0
    mask[:, :, blocked] = 0.0
    if position_allowed:
        for index, allowed in position_allowed.items():
            if sequence[index] == MUTABLE_TOKEN:
                mask[:, index, :] = 1.0
                mask[:, index, _allowed_to_disallowed_indices(allowed)] = 0.0
    return mask


def normalized_gradient_tensor(gradient: Any, gradient_mask: Any) -> Any:
    dependencies = _imports()
    torch = dependencies["torch"]
    masked = gradient * gradient_mask
    mutable_count = (torch.square(masked).sum(-1) > 0).sum(-1)
    norm = torch.linalg.norm(masked, dim=(-1, -2))
    normalized = (masked / (norm[:, None, None] + 1e-7)) * torch.sqrt(mutable_count[:, None, None])
    return normalized * gradient_mask


def design_binder(
    inversion_models: Mapping[str, Any],
    hf_critic_models: Mapping[str, Any],
    esmc_model: Any,
    target_name: str | None = None,
    target_sequence: str | None = None,
    binder_name: str | None = None,
    binder_sequence: str | None = None,
    is_antibody: bool | None = None,
    seed: int = 0,
    batch_size: int = 1,
    allowed_amino_acids: str | None = None,
    *,
    position_allowed: Mapping[int, Iterable[str]] | None = None,
    contact_masks: Any | None = None,
    steps: int,
    learning_rate: float,
    temperature_min: float,
    plm_weight: float,
    plm_weight_antibody: float,
    esmc_mask_fraction: float,
    critic_num_loops: int,
    critic_num_sampling_steps: int,
    loss_weights: Mapping[str, float],
    save_confidence_arrays: bool = False,
) -> tuple[list[str], dict[int, dict[str, list[float]]], list[dict[str, Any]]]:
    """Run the sister algorithm with campaign-resolved loop controls.

    `contact_masks` comes from `runtime.prepare_runtime_request`. The explicit
    argument keeps epitope selection attached to the inter-chain loss.
    """

    if (target_name is None) == (target_sequence is None):
        raise ValueError("provide exactly one of target_name or target_sequence")
    if (binder_name is None) == (binder_sequence is None):
        raise ValueError("provide exactly one of binder_name or binder_sequence")
    if target_name is not None or binder_name is not None:
        raise ValueError("the adapter runtime accepts explicit target and binder sequences")
    if target_sequence is None or binder_sequence is None or is_antibody is None:
        raise ValueError("native design requires target, binder, and antibody state")
    if "|" in binder_sequence:
        raise ValueError("the native binder must be one suffix chain")
    dependencies = _imports()
    torch = dependencies["torch"]
    functional = dependencies["functional"]
    optim = dependencies["optim"]
    target_one_hot = sequence_to_one_hot(target_sequence.replace("|", ""))
    binder_length = len(binder_sequence)
    with dependencies["seed_context"](seed), torch.device("cuda"):
        logits = build_initial_soft_sequence_logits(binder_sequence, batch_size, allowed_amino_acids, position_allowed)
        gradient_mask = build_gradient_mask(binder_sequence, batch_size, allowed_amino_acids, position_allowed).to("cuda")
    if not bool((gradient_mask.sum(dim=-1) > 0).any().item()):
        raise ValueError("native design prompt has no mutable positions")
    optimizer = optim.SGD([logits], lr=learning_rate)
    trajectory: dict[int, dict[str, list[float]]] = {}
    best_sequences = [""] * batch_size
    best_iptm = [-1.0] * batch_size
    for step in range(steps):
        optimizer.zero_grad()
        schedule_position = (step + 1) / steps
        temperature = temperature_min + (1 - temperature_min) * 0.5 * (1 + math.cos(math.pi * schedule_position))
        random.seed(seed + step)
        inversion_model = list(inversion_models.values())[random.randint(0, len(inversion_models) - 1)]
        design = functional.softmax(logits / temperature, dim=-1)
        fold = fold_and_get_distogram(
            inversion_model,
            target_sequence,
            target_one_hot,
            design,
            num_loops=1,
            num_sampling_steps=50 if temperature < 0.05 else 1,
            calculate_confidence=temperature < 0.05,
            seed=seed + step,
        )
        losses = compute_structure_losses(
            fold["distogram_logits"],
            binder_length,
            contact_masks=contact_masks,
            loss_weights=loss_weights,
        )
        structure_loss = losses["total_loss"]
        structure_gradient = torch.autograd.grad(structure_loss.mean(), logits)[0]
        design = functional.softmax(logits / temperature, dim=-1)
        score_mask = gradient_mask.sum(dim=-1) > 0
        with dependencies["seed_context"](seed + step):
            plm_loss = compute_esmc_pseudoperplexity_nll(
                esmc_model,
                design,
                score_mask,
                batch_size=4,
                n_passes=4,
                mask_fraction=esmc_mask_fraction,
            )
        plm_gradient = torch.autograd.grad(plm_loss.mean(), logits)[0]
        logits.grad = normalized_gradient_tensor(structure_gradient, gradient_mask) + (
            plm_weight_antibody if is_antibody else plm_weight
        ) * normalized_gradient_tensor(plm_gradient, gradient_mask)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * temperature
        optimizer.step()
        serial = {key: value.detach().cpu().tolist() for key, value in losses.items()}
        serial["plm_loss"] = plm_loss.detach().cpu().tolist()
        serial["total_loss"] = (structure_loss + plm_loss).detach().cpu().tolist()
        trajectory[step] = serial
        iptm = fold.get("iptm")
        if iptm is not None:
            for index, value in enumerate(iptm):
                score = float(value.item()) if value is not None else -1.0
                if score > best_iptm[index]:
                    best_iptm[index] = score
                    best_sequences[index] = fold["seq_list"][index]
    final_design = functional.softmax(logits / temperature_min, dim=-1)
    fallback_fold = fold_and_get_distogram(
        next(iter(inversion_models.values())),
        target_sequence,
        target_one_hot,
        final_design,
        num_loops=1,
        num_sampling_steps=1,
        calculate_confidence=False,
        seed=seed + steps,
    )
    for index, sequence in enumerate(best_sequences):
        if not sequence:
            best_sequences[index] = fallback_fold["seq_list"][index]
    results: list[dict[str, Any]] = []
    target_length = len(target_sequence.replace("|", ""))
    for batch_index, sequence in enumerate(best_sequences):
        binder = sequence.split("|")[-1]
        binder_design = sequence_to_one_hot(binder)[..., 2:22]
        for critic_name, critic_model in hf_critic_models.items():
            scaling = "ESMFold2-Experimental-Fast-base" in critic_name
            if scaling:
                critic_model.cuda()
            final_fold = fold_and_get_distogram(
                critic_model,
                target_sequence,
                target_one_hot,
                binder_design,
                num_loops=critic_num_loops,
                num_sampling_steps=critic_num_sampling_steps,
                calculate_confidence=True,
                seed=seed,
            )
            if scaling:
                critic_model.cpu()
            iptm = final_fold["iptm"]
            record: dict[str, Any] = {
                "is_antibody": is_antibody,
                "critic_name": critic_name,
                "batch_idx": batch_index,
                "designed_sequence": sequence,
                "complex": complex_to_pdb_string(final_fold["inputs"], final_fold["output"]),
                "final_loss": trajectory[steps - 1]["total_loss"][batch_index],
                "iptm": float(iptm.item()) if iptm is not None else None,
                "logits": logits[batch_index].detach().cpu(),
                **compute_distogram_iptm_proxy(final_fold["distogram_logits"], target_length, binder, is_antibody),
            }
            if save_confidence_arrays:
                record["confidence_arrays"] = {
                    key: value.detach().float().cpu().numpy()
                    for key, value in final_fold["output"].items()
                    if key in {"pae", "plddt", "plddt_ca", "plddt_per_atom", "pde"} and hasattr(value, "detach")
                }
            results.append(record)
    return best_sequences, trajectory, results
