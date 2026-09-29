"""Lazy model loading for the ESMFold2 native-design runtime."""

from __future__ import annotations

from functools import partial
from typing import Any


def _imports() -> dict[str, Any]:
    import torch
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper
    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM, UnifiedTransformerBlock
    from transformers.models.esmfold2.modeling_esmfold2_common import PairUpdateBlock
    from transformers.models.esmfold2.modeling_esmfold2_experimental import ESMFold2ExperimentalModel, MSAEncoder

    return {
        "torch": torch,
        "CheckpointImpl": CheckpointImpl,
        "apply_activation_checkpointing": apply_activation_checkpointing,
        "checkpoint_wrapper": checkpoint_wrapper,
        "ESMCForMaskedLM": ESMCForMaskedLM,
        "TransformerBlock": UnifiedTransformerBlock,
        "PairUpdateBlock": PairUpdateBlock,
        "ESMFold2ExperimentalModel": ESMFold2ExperimentalModel,
        "MSAEncoder": MSAEncoder,
    }


class ESMFold2Designer:
    """Load and retain the inversion models, critics, and ESMC regularizer."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.inversion_models: dict[str, Any] = {}
        self.hf_critic_models: dict[str, Any] = {}
        self.esmc_model: Any = None
        self._esmc: Any = None

    def _load_hf_model(self, repository: str, lm_dropout: float, cache_esmc: bool, device: str) -> Any:
        dependencies = _imports()
        model = dependencies["ESMFold2ExperimentalModel"].from_pretrained(repository, load_esmc=not cache_esmc)
        if cache_esmc:
            if self._esmc is None:
                model.load_esmc(model.config.esmc_id)
                self._esmc = model._esmc
            else:
                model._esmc = self._esmc
        model.configure_lm_dropout(lm_dropout, force_lm_dropout_during_inference=True)
        model.set_kernel_backend(None)
        return model.to(device=device).eval().requires_grad_(False)

    def _apply_torch_compile(self, model: Any) -> None:
        dependencies = _imports()
        torch = dependencies["torch"]
        torch._dynamo.config.cache_size_limit = 512
        torch._dynamo.config.accumulated_cache_size_limit = 512
        targets = (dependencies["MSAEncoder"], dependencies["PairUpdateBlock"], dependencies["TransformerBlock"])

        def compile_module(module: Any) -> None:
            if isinstance(module, targets):
                module.forward = torch.compile(module.forward)

        model.apply(compile_module)

    def load(self, use_scaling_critics: bool) -> None:
        """Load the configured models after the adapter has validated all inputs."""

        config = self.config
        scaling = ()
        if use_scaling_critics:
            scaling = tuple(
                f"biohub/ESMFold2-Experimental-Fast-base{size}-step{step}k"
                for size in ("300M", "600M", "6B")
                for step in ("250", "500", "750", "1000", "1500")
            )
        self.inversion_models = {
            repository: self._load_hf_model(repository, config.lm_dropout_inversion, True, "cuda")
            for repository in config.inversion_repositories
        }
        if config.compile:
            for model in self.inversion_models.values():
                self._apply_torch_compile(model)
        self.hf_critic_models = {
            repository: self._load_hf_model(repository, config.lm_dropout_critic, True, "cuda")
            for repository in config.hero_critic_repositories
        }
        for repository in scaling:
            self.hf_critic_models[repository] = self._load_hf_model(repository, config.lm_dropout_critic, False, "cpu")
        dependencies = _imports()
        torch = dependencies["torch"]
        self.esmc_model = dependencies["ESMCForMaskedLM"].from_pretrained(config.esmc_repository, torch_dtype=torch.float32)
        if config.reuse_esmc:
            del self.esmc_model.esmc
            torch.cuda.empty_cache()
            self.esmc_model.esmc = next(iter(self.inversion_models.values()))._esmc
        self.esmc_model = self.esmc_model.cuda().eval().requires_grad_(False)
        if config.checkpoint_lm:
            dependencies["apply_activation_checkpointing"](
                self.esmc_model,
                checkpoint_wrapper_fn=partial(
                    dependencies["checkpoint_wrapper"], checkpoint_impl=dependencies["CheckpointImpl"].NO_REENTRANT
                ),
                check_fn=lambda module: isinstance(module, dependencies["TransformerBlock"]),
            )
