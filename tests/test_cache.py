import unittest
import weakref

import torch
from transformers import DynamicCache

from orderkv.cache import cache_bytes, compact_cache
from orderkv.positions import LogicalPositions
from orderkv.spans import SourceMap


def positions():
    return LogicalPositions.from_prompt(12, SourceMap(tuple(range(2, 10)),
                                                     (0, 0, 1, 1, 2, 2, 3, 3), frozenset({2, 9})))


def synthetic_cache():
    cache = DynamicCache()
    for layer in range(2):
        values = torch.arange(384, dtype=torch.float32).reshape(1, 2, 12, 16) + layer
        cache.update(values, values.clone(), layer)
    return cache


class CacheTests(unittest.TestCase):
    def test_middle_removal_independent_storage_and_logical_clock(self):
        cache = synthetic_cache()
        old_keys = list(cache.key_cache)
        keep = tuple(i for i in range(12) if i not in (4, 5))
        result = compact_cache(cache, keep, positions())
        self.assertIs(result.cache, cache)
        self.assertEqual(result.positions.kept, keep)
        self.assertEqual(result.positions.next_position, 12)
        self.assertEqual(result.after.storage, result.before.storage * 10 // 12)
        self.assertEqual(result.after.storage, result.after.tensors)
        for old, new in zip(old_keys, cache.key_cache):
            torch.testing.assert_close(new, old[:, :, keep, :])
            self.assertNotEqual(new.untyped_storage().data_ptr(), old.untyped_storage().data_ptr())
        repeated = compact_cache(cache, range(10), result.positions)
        self.assertEqual(repeated.before, repeated.after)

    def test_keep_all_is_exact_noop(self):
        cache = synthetic_cache()
        pointers = [t.data_ptr() for t in cache.key_cache]
        compact_cache(cache, range(12), positions())
        self.assertEqual(pointers, [t.data_ptr() for t in cache.key_cache])

    def test_protection_and_validation_before_mutation(self):
        for keep in ((0, 0), (3, 2), (-1,), (12,), (1.5,), tuple(range(1, 12)),
                     tuple(i for i in range(12) if i != 2)):
            cache = synthetic_cache()
            before = cache_bytes(cache)
            with self.assertRaises(ValueError):
                compact_cache(cache, keep, positions())
            self.assertEqual(cache_bytes(cache), before)
        state = positions().append()
        cache = synthetic_cache()
        for layer in range(2):
            cache.update(torch.zeros(1, 2, 1, 16), torch.zeros(1, 2, 1, 16), layer)
        with self.assertRaises(ValueError):
            compact_cache(cache, range(12), state)

    def test_storage_deduplicates_views(self):
        cache = DynamicCache()
        backing = torch.zeros(1, 2, 12, 16)
        cache.update(backing[:, :, :6], backing[:, :, 6:], 0)
        self.assertEqual(cache_bytes(cache).storage, backing.untyped_storage().nbytes())

    def test_old_tensors_released_and_malformed_layers_rejected(self):
        cache = synthetic_cache()
        old = [weakref.ref(t) for t in (*cache.key_cache, *cache.value_cache)]
        compact_cache(cache, tuple(i for i in range(12) if i not in (4, 5)), positions())
        self.assertTrue(all(ref() is None for ref in old))
        cache = synthetic_cache()
        cache.value_cache[1] = cache.value_cache[1][:, :, :8]
        first = cache.key_cache[0]
        with self.assertRaises(ValueError):
            compact_cache(cache, (0, 1, 2, 3, 6, 7, 8, 9, 10, 11), positions())
        self.assertIs(first, cache.key_cache[0])
