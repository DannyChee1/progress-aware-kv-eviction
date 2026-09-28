import unittest

from orderkv.spans import map_source


class SpanTests(unittest.TestCase):
    def test_unicode_escapes_and_special_offsets(self):
        prompt = 'SYS|café "a\\b"|END'
        offsets = [(0, 0)] + [(i, i + 1) for i in range(len(prompt))] + [(0, 0)]
        source = map_source(offsets, (4, len(prompt) - 4), block_tokens=2,
                            protect_first=0, protect_last=0)
        self.assertEqual(source.positions, tuple(range(5, len(prompt) - 3)))
        self.assertTrue({0, 1, 2, 3, 4, len(offsets) - 1} <= source.protected_positions)
        self.assertTrue({source.positions[0], source.positions[-1]} <= source.protected_positions)

    def test_straddling_token_protects_entire_block(self):
        offsets = [(0, 5), (5, 7), (7, 9), (9, 11), (11, 15)]
        source = map_source(offsets, (4, 12), block_tokens=2, protect_first=0, protect_last=0)
        self.assertEqual(source.protected_positions, frozenset({0, 1, 4}))
        self.assertEqual(source.positions, (0, 1, 2, 3, 4))

    def test_prompt_protection_expands_to_blocks(self):
        source = map_source([(i, i + 1) for i in range(20)], (2, 18),
                            block_tokens=4, protect_first=7, protect_last=3)
        self.assertTrue(set(range(10)) <= source.protected_positions)
        self.assertTrue(set(range(14, 20)) <= source.protected_positions)
