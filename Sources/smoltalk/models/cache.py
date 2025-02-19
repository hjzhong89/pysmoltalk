from typing import Any, Optional, Sequence

import torch
from transformers.cache_utils import Cache


class SliceUpdateKeyValueCache(Cache):
    """
    Helper class for in-place slice updating key/value caches.

    Reference: https://machinelearning.apple.com/research/core-ml-on-device-llama
    """

    def __init__(
            self,
            *,
            shape: Sequence[int],
            device: torch.device,
            dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create key/value cache of shape:
        (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)

    def update(
            self,
            k_state: torch.Tensor,
            v_state: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key / value cache tensors for slice [begin, end).
        Return slice of key / value cache tensors from [0, end)."""
        position = cache_kwargs.get("cache_position", None)
        assert position is not None, "cache_position required to update cache."
        begin, end = self.past_seen_tokens, self.past_seen_tokens + position.shape[-1]
        self.k[layer_idx, :, : k_state.shape[1], begin:end, :] = k_state
        self.v[layer_idx, :, : v_state.shape[1], begin:end, :] = v_state
        k_state = self.k[layer_idx, :, :, :end, :]
        v_state = self.v[layer_idx, :, :, :end, :]
        return k_state, v_state

    def get_seq_length(self, _: int = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
        return None
