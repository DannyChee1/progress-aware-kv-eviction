from dataclasses import dataclass, replace

import torch

from orderkv.spans import SourceMap


@dataclass(frozen=True)
class LogicalPositions:
    kept: tuple[int, ...]
    next_position: int
    prompt_length: int
    protected: frozenset[int]
    source_blocks: tuple[int | None, ...]

    def __post_init__(self):
        if not 0 <= self.prompt_length <= self.next_position:
            raise ValueError("Invalid prompt or next logical position")
        if len(self.source_blocks) != self.prompt_length:
            raise ValueError("Source ownership must cover the original prompt")
        if tuple(sorted(set(self.kept))) != self.kept:
            raise ValueError("Logical positions must be unique and increasing")
        if any(p < 0 or p >= self.next_position for p in self.kept):
            raise ValueError("Kept positions must precede the next position")
        if any(p < 0 or p >= self.prompt_length for p in self.protected):
            raise ValueError("Protected prompt position out of bounds")
        required = self.protected | frozenset(range(self.prompt_length, self.next_position))
        required |= frozenset(i for i, block in enumerate(self.source_blocks) if block is None)
        if not required <= set(self.kept):
            raise ValueError("Instructions, protected rows, and generated rows must remain")

    @classmethod
    def from_prompt(cls, prompt_length: int, source: SourceMap):
        if len(source.positions) != len(source.block_ids):
            raise ValueError("Source positions and block IDs disagree")
        if len(set(source.positions)) != len(source.positions):
            raise ValueError("Duplicate source positions")
        blocks = [None] * prompt_length
        for position, block in zip(source.positions, source.block_ids):
            if not 0 <= position < prompt_length or block < 0:
                raise ValueError("Invalid source position or block")
            blocks[position] = block
        return cls(tuple(range(prompt_length)), prompt_length, prompt_length,
                   source.protected_positions, tuple(blocks))

    def append(self):
        return replace(self, kept=self.kept + (self.next_position,), next_position=self.next_position + 1)


def causal_mask(query_positions, key_positions, *, dtype, device, blocked=frozenset()):
    if not dtype.is_floating_point:
        raise ValueError("Attention mask must have a floating dtype")
    queries = torch.tensor(query_positions, dtype=torch.long, device=device)
    keys = torch.tensor(key_positions, dtype=torch.long, device=device)
    allowed = keys[None, :] <= queries[:, None]
    if blocked:
        allowed &= ~torch.isin(keys, torch.tensor(sorted(blocked), device=device))[None, :]
    if not bool(allowed.any(dim=-1).all()):
        raise ValueError("Every query must have at least one visible key")
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None, :, :]
