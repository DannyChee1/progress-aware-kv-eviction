"""Case types: the public part a model sees and the private gold answer."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    description: str


@dataclass(frozen=True)
class CasePublic:
    case_id: str
    split: str
    document: str
    fields: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class GoldPrivate:
    values: dict[str, str]
    answer_spans: dict[str, tuple[tuple[int, int], ...]]
    support_spans: dict[str, tuple[tuple[int, int], ...]]
