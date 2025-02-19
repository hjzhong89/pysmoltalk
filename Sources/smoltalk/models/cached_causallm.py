import torch
from transformers import Qwen2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from smoltalk.models.cache import SliceUpdateKeyValueCache


class KvCacheStateLlamaForCausalLM(torch.nn.Module):
    """
    Model wrapper to swap cache implementation and register as buffers.

    Reference: https://machinelearning.apple.com/research/core-ml-on-device-llama
    """

    def __init__(
            self,
            model_path: str,
            *,
            batch_size: int = 1,
            context_size: int = 4096,
            device: str = "mps"
    ) -> None:
        super().__init__()
        self.model = Qwen2ForCausalLM.from_pretrained(model_path).to(device)
        self.config = self.model.config

        self.kv_cache_shape: tuple[int, ...] = (
            self.config.num_hidden_layers,
            batch_size,
            self.config.num_key_value_heads,
            context_size,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        # Register KV cache buffers to be recognized as Core ML states
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape, device=device)
        self.register_buffer("keyCache", self.kv_cache.k)
        self.register_buffer("valueCache", self.kv_cache.v)

    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.LongTensor,
            causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits
