from dataclasses import dataclass, replace
from numbers import Integral

import torch
from transformers import DynamicCache

from orderkv.positions import LogicalPositions


@dataclass(frozen=True)
class CacheBytes:
    storage: int
    tensors: int


def cache_bytes(cache: DynamicCache) -> CacheBytes:
    storages = {}
    tensors = 0
    for tensor in (*cache.key_cache, *cache.value_cache):
        storage = tensor.untyped_storage()
        storages[(str(tensor.device), storage.data_ptr())] = storage.nbytes()
        tensors += tensor.numel() * tensor.element_size()
    return CacheBytes(sum(storages.values()), tensors)


@dataclass(frozen=True)
class CompactResult:
    cache: DynamicCache
    positions: LogicalPositions
    before: CacheBytes
    after: CacheBytes


@torch.inference_mode()
def compact_cache(cache: DynamicCache, keep_physical_indices,
                  logical_positions: LogicalPositions) -> CompactResult:
    if type(cache) is not DynamicCache:
        raise TypeError("Only the pinned DynamicCache implementation is supported")
    keep = tuple(keep_physical_indices)
    if any(not isinstance(i, Integral) or isinstance(i, bool) for i in keep):
        raise ValueError("Physical indices must be integers")
    if tuple(sorted(set(keep))) != keep:
        raise ValueError("Physical indices must be unique and increasing")
    length = len(logical_positions.kept)
    if any(i < 0 or i >= length for i in keep):
        raise ValueError("Physical index out of bounds")
    if not cache.key_cache or len(cache.key_cache) != len(cache.value_cache):
        raise ValueError("Cache must contain matching populated K/V layers")
    for key, value in zip(cache.key_cache, cache.value_cache):
        if key.ndim != 4 or key.shape != value.shape or key.shape[-2] != length:
            raise ValueError("Every K/V layer must match the physical position map")
        if key.shape[0] != 1:
            raise ValueError("The pilot supports batch size one")
    updated = replace(logical_positions, kept=tuple(logical_positions.kept[i] for i in keep))
    before = cache_bytes(cache)
    if keep == tuple(range(length)):
        return CompactResult(cache, updated, before, before)
    keys, values = [], []
    for key, value in zip(cache.key_cache, cache.value_cache):
        indices = torch.tensor(keep, dtype=torch.long, device=key.device)
        keys.append(key.index_select(-2, indices).contiguous())
        values.append(value.index_select(-2, indices.to(value.device)).contiguous())
    # Preserve logical history; DynamicCache's row count is only physical length.
    cache.key_cache, cache.value_cache = keys, values
    cache._seen_tokens = updated.next_position
    return CompactResult(cache, updated, before, cache_bytes(cache))
