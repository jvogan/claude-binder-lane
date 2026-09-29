"""Loss functions for native ESMFold2 design.

Every tensor dependency is imported when a loss executes. This module therefore
remains importable in the adapter's CPU-only contract environment.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any, Iterable, Mapping

from .constants import ESMC_MASK_FRACTION, LOSS_WEIGHTS, PROTEIN_1TO3, TOKENS


def _torch() -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional

    return torch, functional


def get_mid_points() -> Any:
    """Return the 128 ESMFold2 distogram midpoint values."""

    torch, _ = _torch()
    boundaries = torch.linspace(2, 52.0, 127)
    return (torch.cat((torch.tensor([1.0]), boundaries, torch.tensor([57.0])))[:-1]
            + torch.cat((torch.tensor([1.0]), boundaries, torch.tensor([57.0])))[1:]) / 2


def binned_entropy(distogram_logits: Any, bin_distance: Any, cutoff: float) -> Any:
    torch, _ = _torch()
    masked_logits = distogram_logits - (1e7 * ~(bin_distance < cutoff))
    probabilities = torch.softmax(masked_logits, dim=-1)
    return -(probabilities * torch.log_softmax(distogram_logits, dim=-1)).sum(-1)


def masked_min_k(values: Any, mask: Any, k: int) -> Any:
    torch, _ = _torch()
    selected = torch.sort(torch.where(mask.bool(), values, float("nan")))[0]
    ranks = torch.arange(selected.shape[-1], device=selected.device) < k
    ranks = ranks & ~torch.isnan(selected)
    return torch.where(ranks, selected, 0).sum(-1) / (ranks.sum(-1) + 1e-8)


def masked_average(values: Any, mask: Any) -> Any:
    torch, _ = _torch()
    selected = mask.bool()
    return torch.where(selected, values, 0).sum(-1) / (torch.where(selected, 1, 0).sum(-1) + 1e-8)


def _mask_tensor(mask: Iterable[bool], device: Any) -> Any:
    torch, _ = _torch()
    return torch.tensor(tuple(mask), device=device, dtype=torch.bool)


def compute_contact_loss(
    distogram_logits: Any,
    bin_distance: Any,
    num_contacts: int,
    min_sep: int,
    cutoff: float,
    chain_mask: Any,
    binder_mask: Any,
) -> Any:
    """Compute the sister implementation's masked contact-entropy loss."""

    torch, _ = _torch()
    contact_loss = binned_entropy(distogram_logits, bin_distance, cutoff)
    positions = torch.arange(distogram_logits.shape[1], device=distogram_logits.device)
    if min_sep > 0:
        separation = torch.abs(positions[:, None] - positions[None, :]) >= min_sep
        binder_mask = torch.logical_and(separation, binder_mask)
    per_residue = masked_min_k(contact_loss, binder_mask, k=num_contacts)
    return masked_average(per_residue, chain_mask)


def compute_intra_contact_loss(distogram_logits: Any, binder_length: int, bin_distance: Any) -> Any:
    torch, _ = _torch()
    length = distogram_logits.shape[1]
    binder = torch.zeros(length, device=distogram_logits.device, dtype=torch.bool)
    binder[-binder_length:] = True
    return compute_contact_loss(distogram_logits, bin_distance, 2, 9, 14.0, binder, binder)


def compute_inter_contact_loss(
    distogram_logits: Any,
    binder_length: int,
    bin_distance: Any,
    *,
    chain_mask: Iterable[bool] | Any | None = None,
    binder_mask: Iterable[bool] | Any | None = None,
) -> Any:
    """Compute the interface loss with an optional epitope target mask.

    The `chain_mask` selects target rows for the averaging axis. Passing the
    contact mask prepared from epitope residues makes the aim part of the loss.
    """

    torch, _ = _torch()
    length = distogram_logits.shape[1]
    default_binder = torch.zeros(length, device=distogram_logits.device, dtype=torch.bool)
    default_binder[-binder_length:] = True
    if binder_mask is None:
        binder = default_binder
    elif isinstance(binder_mask, torch.Tensor):
        binder = binder_mask.to(device=distogram_logits.device, dtype=torch.bool)
    else:
        binder = _mask_tensor(binder_mask, distogram_logits.device)
    if chain_mask is None:
        chain = ~default_binder
    elif isinstance(chain_mask, torch.Tensor):
        chain = chain_mask.to(device=distogram_logits.device, dtype=torch.bool)
    else:
        chain = _mask_tensor(chain_mask, distogram_logits.device)
    if chain.numel() != length or binder.numel() != length:
        raise ValueError("contact masks must match the folded complex length")
    return compute_contact_loss(distogram_logits, bin_distance, 1, 0, 22.0, chain, binder)


def compute_globularity_loss(distogram_logits: Any, binder_length: int, bin_distance: Any) -> Any:
    torch, functional = _torch()
    binder_disto = distogram_logits[:, -binder_length:, -binder_length:, :]
    count = binder_disto.shape[1]
    probabilities = torch.softmax(binder_disto, dim=-1)
    expected_squared_distance = (probabilities * torch.square(bin_distance.clamp(max=27))).sum(-1)
    radius = torch.sqrt(torch.tril(expected_squared_distance, diagonal=-1).sum(dim=(1, 2)) / (count * count))
    return functional.elu(radius - 2.38 * (count**0.365))


def compute_structure_losses(
    distogram_logits: Any,
    binder_length: int,
    *,
    contact_masks: Any | None = None,
    loss_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compute native structure losses and thread prepared aiming masks through."""

    torch, _ = _torch()
    bin_distance = get_mid_points().to(distogram_logits.device)
    weights = dict(LOSS_WEIGHTS if loss_weights is None else loss_weights)
    chain_mask = None if contact_masks is None else contact_masks.chain_mask
    binder_mask = None if contact_masks is None else contact_masks.binder_mask
    losses = {
        "intra_contact_loss": compute_intra_contact_loss(distogram_logits, binder_length, bin_distance),
        "inter_contact_loss": compute_inter_contact_loss(
            distogram_logits,
            binder_length,
            bin_distance,
            chain_mask=chain_mask,
            binder_mask=binder_mask,
        ),
        "glob_loss": compute_globularity_loss(distogram_logits, binder_length, bin_distance),
    }
    batch_size = distogram_logits.size(0)
    total = torch.zeros(batch_size, device=distogram_logits.device, requires_grad=True)
    total = total + weights["intra_contact"] * losses["intra_contact_loss"]
    total = total + weights["inter_contact"] * losses["inter_contact_loss"]
    total = total + weights["glob"] * losses["glob_loss"]
    losses["total_loss"] = total
    return losses


def _binding_confidence_entropy(distogram: Any, bin_distance: Any, cutoff: float) -> Any:
    torch, _ = _torch()
    probabilities = torch.softmax(distogram, dim=-1)
    selected = probabilities[..., bin_distance < cutoff]
    selected = selected / (selected.sum(-1, keepdim=True) + 1e-8)
    return -(selected * torch.log(selected + 1e-10)).sum(-1)


def compute_distogram_iptm_proxy(
    distogram_logits: Any,
    target_length: int,
    binder_sequence: str,
    is_antibody: bool,
) -> dict[str, float]:
    """Return the sister design's distogram iPTM proxy values."""

    torch, _ = _torch()
    if distogram_logits.ndim == 4:
        distogram_logits = distogram_logits[0]
    binder_length = len(binder_sequence)
    if distogram_logits.shape[0] != target_length + binder_length:
        raise ValueError("distogram length differs from target plus binder length")
    entropy = _binding_confidence_entropy(
        distogram_logits[target_length:, :target_length, :],
        get_mid_points().to(distogram_logits.device),
        22.0,
    )
    count = min(binder_length, entropy.numel())
    mean = float(torch.sort(entropy.reshape(-1)).values[:count].mean())
    score = float(max(0.0, min(1.0, 1.0 - mean / math.log(51))))
    cdr_score = float("nan")
    if is_antibody:
        try:
            from abnumber import Chain

            chain = Chain(binder_sequence, scheme="chothia")
            cdrs = [chain.cdr1_seq, chain.cdr2_seq, chain.cdr3_seq]
            cdr_indices = [
                index
                for cdr in cdrs
                for index in range(binder_sequence.find(cdr), binder_sequence.find(cdr) + len(cdr))
                if index >= 0
            ]
            if cdr_indices:
                cdr_entropy = _binding_confidence_entropy(
                    distogram_logits[[target_length + index for index in cdr_indices], :target_length, :],
                    get_mid_points().to(distogram_logits.device),
                    22.0,
                )
                cdr_mean = float(torch.sort(cdr_entropy.reshape(-1)).values[: len(cdr_indices)].mean())
                cdr_score = float(max(0.0, min(1.0, 1.0 - cdr_mean / math.log(51))))
        except ImportError:
            pass
    return {"distogram_iptm_proxy": score, "cdr_distogram_iptm_proxy": cdr_score}


@cache
def _folding_trunk_to_lm_aa_vocab_matrix(device_name: str) -> Any:
    torch, _ = _torch()
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

    three_to_one = {value: key for key, value in PROTEIN_1TO3.items()}
    folding_aas = [three_to_one[token] for token in TOKENS[2:22]]
    lm_aas = [value[0] for value in sorted(ESMCTokenizer().vocab.items(), key=lambda value: value[1])[4:24]]
    matrix = torch.zeros(20, 20)
    for index, residue in enumerate(folding_aas):
        matrix[index, lm_aas.index(residue)] = 1
    return matrix.to(device=device_name)


def compute_esmc_pseudoperplexity_nll(
    esmc_model: Any,
    binder_design: Any,
    score_mask: Any,
    batch_size: int = 4,
    n_passes: int = 4,
    mask_fraction: float = ESMC_MASK_FRACTION,
) -> Any:
    """Compute the ESMC pseudoperplexity regularizer used by the sister loop."""

    torch, functional = _torch()
    from transformers.models.esmc.tokenization_esmc import ESMCTokenizer

    device = binder_design.device
    target_esm = binder_design @ _folding_trunk_to_lm_aa_vocab_matrix(str(device))
    discrete = functional.one_hot(torch.argmax(target_esm, dim=-1), num_classes=target_esm.size(-1)).to(target_esm.dtype)
    input_esm = target_esm + (discrete - target_esm).detach()
    vocabulary_size = esmc_model.config.vocab_size
    dtype = esmc_model.esmc.embed.weight.dtype
    input_ids = torch.zeros((binder_design.size(0), binder_design.size(1) + 2, vocabulary_size), dtype=dtype, device=device)
    tokenizer = ESMCTokenizer()
    input_ids[:, 0, tokenizer.cls_token_id] = 1
    input_ids[:, -1, tokenizer.eos_token_id] = 1
    input_ids[:, 1:-1, 4:24] = input_esm.to(dtype)
    if score_mask.ndim == 1:
        score_mask = score_mask.unsqueeze(0).expand(binder_design.size(0), -1)
    if score_mask.shape != binder_design.shape[:2]:
        raise ValueError("ESMC score mask shape differs from the binder design")
    score_mask = score_mask.to(device=device, dtype=torch.bool)
    mask_token = torch.zeros(vocabulary_size, dtype=dtype, device=device)
    mask_token[esmc_model.config.mask_token_id] = 1
    losses = []
    for index in range(binder_design.size(0)):
        positions = score_mask[index].nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            raise ValueError("ESMC pseudoperplexity score mask selected zero positions")
        count = max(1, math.ceil(mask_fraction * int(positions.numel())))
        offsets = torch.rand((n_passes, positions.numel()), device=device).topk(count, dim=-1, largest=False).indices
        pass_masks = torch.zeros((n_passes, binder_design.size(1)), dtype=torch.bool, device=device)
        pass_masks[torch.arange(n_passes, device=device)[:, None], positions[offsets]] = True
        sequences = input_ids[index:index + 1].repeat(n_passes, 1, 1)
        rows, columns = pass_masks.nonzero(as_tuple=True)
        sequences[rows, columns + 1] = mask_token
        nlls = []
        for start in range(0, n_passes, batch_size):
            stop = min(start + batch_size, n_passes)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden, *_ = esmc_model.esmc.transformer(
                    sequences[start:stop] @ esmc_model.esmc.embed.weight.to(sequences.dtype),
                    sequence_id=None,
                    layers_to_collect=[],
                    output_attentions=False,
                )
                logits = esmc_model.lm_head(hidden)
            log_probabilities = logits.log_softmax(dim=-1)[:, 1:-1, 4:24]
            values = -(log_probabilities * target_esm[index].to(log_probabilities.dtype).unsqueeze(0)).sum(dim=-1)
            nlls.append(values[pass_masks[start:stop]])
        losses.append(torch.cat(nlls).mean())
    return torch.stack(losses)
