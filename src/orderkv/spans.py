from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SourceMap:
    positions: tuple[int, ...]
    block_ids: tuple[int, ...]
    protected_positions: frozenset[int]


def overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def map_source(offsets: Sequence[tuple[int, int]], source_span: tuple[int, int],
               block_tokens: int = 128, protect_first: int = 32,
               protect_last: int = 64) -> SourceMap:
    start, end = source_span
    if start < 0 or end <= start or block_tokens < 1 or min(protect_first, protect_last) < 0:
        raise ValueError("Invalid span or protection settings")
    if any(a < 0 or b < a for a, b in offsets):
        raise ValueError("Invalid tokenizer offsets")
    positions = tuple(i for i, span in enumerate(offsets) if overlaps(span, source_span))
    blocks = tuple(i // block_tokens for i in range(len(positions)))
    protected = set(range(len(offsets))) - set(positions)
    protected.update(range(min(protect_first, len(offsets))))
    protected.update(range(max(0, len(offsets) - protect_last), len(offsets)))
    for i in positions:
        a, b = offsets[i]
        if a <= start or b >= end:
            protected.add(i)
    boundary_blocks = {block for pos, block in zip(positions, blocks) if pos in protected}
    protected.update(pos for pos, block in zip(positions, blocks) if block in boundary_blocks)
    return SourceMap(positions, blocks, frozenset(protected))
