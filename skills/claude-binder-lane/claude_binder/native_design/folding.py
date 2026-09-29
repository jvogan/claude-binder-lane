"""Lazy ESMFold2 feature preparation, folding, and PDB serialization."""

from __future__ import annotations

from functools import cache
from typing import Any

from .constants import PROTEIN_3TO1, TOKENS


def _imports() -> dict[str, Any]:
    """Load the ESMFold2 dependency set only when a fold is requested."""

    import biotite.structure
    import numpy
    import torch
    import torch.nn.functional as functional
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput, load_ccd, prepare_esmfold2_input
    from esm.models.esmfold2.constants import MOL_TYPE_NONPOLYMER
    from esm.utils.structure.protein_chain import ProteinChain
    from esm.utils.structure.protein_complex import ProteinComplex
    from transformers.models.esmfold2.modeling_esmfold2_common import _seed_context

    return {
        "biotite": biotite.structure,
        "numpy": numpy,
        "torch": torch,
        "functional": functional,
        "ProteinInput": ProteinInput,
        "StructurePredictionInput": StructurePredictionInput,
        "load_ccd": load_ccd,
        "prepare_esmfold2_input": prepare_esmfold2_input,
        "MOL_TYPE_NONPOLYMER": MOL_TYPE_NONPOLYMER,
        "ProteinChain": ProteinChain,
        "ProteinComplex": ProteinComplex,
        "seed_context": _seed_context,
    }


_ATOM_FEATURE_DIMS = {
    "ref_pos": 0,
    "ref_element": 0,
    "ref_charge": 0,
    "ref_atom_name_chars": 0,
    "ref_space_uid": 0,
    "atom_attention_mask": 0,
    "atom_to_token": 0,
    "is_resolved": 0,
    "gt_coords": 1,
}


def _resize_tensor(value: Any, *, dim: int, size: int, torch: Any) -> Any:
    if value.shape[dim] >= size:
        return value.narrow(dim, 0, size)
    shape = list(value.shape)
    shape[dim] = size - value.shape[dim]
    padding = torch.zeros(shape, dtype=value.dtype, device=value.device)
    return torch.cat((value, padding), dim=dim)


@cache
def _ensure_ccd_loaded() -> None:
    _imports()["load_ccd"]()


def prepare_esmfold2_tensors(
    input_value: Any,
    max_tokens: int | None = None,
    max_atoms: int | None = None,
    max_seqs: int = 16384,
    pad_to_max_seqs: bool = False,
    seed: int | None = None,
    use_vectorized_msa_assembly: bool = True,
) -> dict[str, Any]:
    """Prepare ESMFold2 features with the sister function signature."""

    del max_tokens, max_seqs, pad_to_max_seqs, use_vectorized_msa_assembly
    dependencies = _imports()
    _ensure_ccd_loaded()
    features, _ = dependencies["prepare_esmfold2_input"](input_value, seed=seed)
    if max_atoms is not None:
        for key, dimension in _ATOM_FEATURE_DIMS.items():
            if key in features:
                features[key] = _resize_tensor(features[key], dim=dimension, size=max_atoms, torch=dependencies["torch"])
    return features


def fold_and_get_distogram(
    model: Any,
    target_seq: str,
    target_one_hot: Any,
    design: Any,
    num_loops: int = 0,
    num_sampling_steps: int = 1,
    calculate_confidence: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run ESMFold2 with a differentiable soft binder sequence."""

    dependencies = _imports()
    torch = dependencies["torch"]
    functional = dependencies["functional"]
    padded_design = functional.pad(design, (2, 11), mode="constant", value=0)
    token_lists = torch.argmax(padded_design, dim=-1)
    designed_sequences = [
        "".join(PROTEIN_3TO1[TOKENS[int(token.item())]] for token in tokens)
        for tokens in token_lists
    ]
    sequence_list = [target_seq + "|" + sequence for sequence in designed_sequences]
    max_atoms = None if len(sequence_list) == 1 else ((len(sequence_list[0]) - 1) * 14) // 32 * 32
    input_list = []
    for sequence in sequence_list:
        chains = sequence.split("|")
        raw_input = dependencies["StructurePredictionInput"](
            sequences=[dependencies["ProteinInput"](id=str(index), sequence=chain, msa=None) for index, chain in enumerate(chains)]
        )
        input_list.append(prepare_esmfold2_tensors(raw_input, max_atoms=max_atoms, seed=seed))
    inputs = {key: torch.stack([value[key] for value in input_list], dim=0).cuda() for key in input_list[0]}
    inputs["res_type_soft"] = torch.cat((target_one_hot.repeat(design.size(0), 1, 1), padded_design), dim=1)
    with dependencies["seed_context"](seed):
        output = model(
            **inputs,
            num_diffusion_samples=1,
            num_sampling_steps=num_sampling_steps,
            num_loops=num_loops,
            calculate_confidence=calculate_confidence,
            seed=seed,
        )
    result = {
        "distogram_logits": output["distogram_logits"],
        "inputs": inputs,
        "inputs_list": input_list,
        "output": output,
        "seq_list": sequence_list,
    }
    if calculate_confidence:
        result.update({"ptm": output.get("ptm"), "iptm": output.get("iptm"), "plddt": output.get("plddt")})
    return result


_CHAIN_ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _chain_label(index: int) -> str:
    if index < 0:
        raise ValueError("chain index must be nonnegative")
    result = ""
    while True:
        result = _CHAIN_ID_ALPHABET[index % len(_CHAIN_ID_ALPHABET)] + result
        index = index // len(_CHAIN_ID_ALPHABET) - 1
        if index < 0:
            return result


def build_complex(inputs: dict[str, Any], output: dict[str, Any]) -> Any:
    """Reconstruct the sister `ProteinComplex` value from ESMFold2 output."""

    dependencies = _imports()
    biotite = dependencies["biotite"]
    atoms = []
    values = zip(
        output["sample_atom_coords"][0].detach().cpu().numpy(),
        inputs["atom_to_token"][0].detach().cpu().numpy(),
        inputs["ref_atom_name_chars"][0].detach().cpu().numpy(),
        inputs["ref_element"][0].detach().cpu().numpy(),
        inputs["atom_attention_mask"][0].detach().cpu().numpy(),
    )
    token_residue = inputs["res_type"][0].detach().cpu().numpy()
    token_index = inputs["token_index"][0].detach().cpu().numpy()
    asym_id = inputs["asym_id"][0].detach().cpu().numpy()
    mol_type = inputs["mol_type"][0].detach().cpu().numpy()
    for coordinates, token, name_chars, element, present in values:
        if not present:
            continue
        token_index_value = int(token)
        atoms.append(
            biotite.Atom(
                coord=coordinates,
                chain_id=_chain_label(int(asym_id[token_index_value])),
                res_id=int(token_index[token_index_value]) + 1,
                res_name=TOKENS[int(token_residue[token_index_value])],
                atom_name="".join(chr(int(character) + 32) for character in name_chars if character != 0),
                element=str(element),
                ins_code=" ",
                hetero=mol_type[token_index_value] == dependencies["MOL_TYPE_NONPOLYMER"],
                b_factor=0.0,
            )
        )
    atom_array = biotite.array(atoms)
    return dependencies["ProteinComplex"].from_chains(
        [dependencies["ProteinChain"].from_atomarray(chain) for chain in biotite.chain_iter(atom_array)]
    )


def complex_to_pdb_string(inputs: dict[str, Any], output: dict[str, Any]) -> str:
    """Serialize a model output to PDB text within the guarded GPU boundary."""

    return build_complex(inputs, output).to_pdb_string()
