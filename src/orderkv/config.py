from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import yaml

CONDITIONS = {'F0', 'R0', 'M0', 'L0', 'S0', 'S1', 'T0', 'T1'}
MODELS = ('Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct')


def series_id(values: dict) -> str:
    """Name of the results directory for a dispatched config: SHA-256 of its fields."""
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    experiment: str = 'longproc'
    model_id: str = 'Qwen/Qwen2.5-7B-Instruct'
    model_revision: str = 'FROM_PREPARATION_MANIFEST'
    dtype: str = 'bfloat16'
    attention_backend: str = 'sdpa'
    device: str = 'cuda'
    batch_size: int = 1
    seed: int = 0
    context_targets: tuple[int, ...] = (16384,)
    context_tolerance: float = 0.6
    max_new_tokens: int = 1024
    block_tokens: int = 128
    protect_prompt_first: int = 32
    protect_prompt_last: int = 64
    cursor_margin_blocks: int = 8
    baseline_budget_path: str = ''
    conditions: tuple[str, ...] = ('F0', 'R0')
    dataset: str = 'data/longproc-0.5k/dev_public.jsonl'
    private_labels: str = 'data/longproc-0.5k/dev_private.jsonl'
    case_limit: int = 48
    shard_seconds: int = 1650
    timeout_seconds: int = 1800

    def validate(self):
        defaults = Config()
        for name in self.__dataclass_fields__:
            expected = type(getattr(defaults, name))
            value = getattr(self, name)
            if expected is tuple:
                if not isinstance(value, (tuple, list)):
                    raise ValueError(f'{name} must be a sequence')
            elif expected is float:
                if type(value) not in (int, float):
                    raise ValueError(f'{name} must be numeric')
            elif type(value) is not expected:
                raise ValueError(f'{name} must be {expected.__name__}')
        if self.model_id not in MODELS:
            raise ValueError(f'model_id must be one of {MODELS}')
        if not re.fullmatch(r'[a-f0-9]{40}', self.model_revision):
            raise ValueError('Resolve the immutable model revision before validation or dispatch')
        if not self.conditions or not set(self.conditions) <= CONDITIONS or len(set(self.conditions)) != len(self.conditions):
            raise ValueError(f'conditions must be distinct members of {sorted(CONDITIONS)}')
        if self.cursor_margin_blocks < 0:
            raise ValueError('Cursor margin must be nonnegative')
        if self.batch_size != 1 or self.device != 'cuda' or self.dtype != 'bfloat16':
            raise ValueError('Runs use batch size 1, CUDA and BF16')
        if self.attention_backend not in ('eager', 'sdpa'):
            raise ValueError('Unsupported attention backend')
        if not 0 < self.shard_seconds <= 1650 or not self.shard_seconds < self.timeout_seconds <= 1800:
            raise ValueError('Invalid bounded shard timing')
        if not 1 <= self.case_limit <= 48 or not 1 <= self.max_new_tokens <= 8192 or self.block_tokens not in (32, 64, 128):
            raise ValueError('Case, token or block limits exceeded')
        if self.protect_prompt_first < 32 or self.protect_prompt_last < 64:
            raise ValueError('Invalid retention boundaries')
        if not self.context_targets or not set(self.context_targets) <= {2048, 4096, 8192, 16384}:
            raise ValueError('Invalid context targets')
        if not 0.05 <= self.context_tolerance <= 0.95:
            raise ValueError('Invalid context tolerance')
        return self

    def digest(self):
        return series_id(asdict(self))


def load_config(path):
    values = yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError('Config must be a mapping')
    for key in ('context_targets', 'conditions'):
        if key in values:
            values[key] = tuple(values[key])
    try:
        return Config(**values).validate()
    except TypeError as error:
        raise ValueError(f'Unknown or invalid configuration: {error}') from error
