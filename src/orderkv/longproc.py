"""LongProc HTML-to-TSV records as cases: the HTML page is the document, and the single field carries the
task-specific tail of LongProc's own prompt (target information and output format, header included)."""
from pathlib import Path

from orderkv.data import CasePublic, FieldSpec, GoldPrivate


def task_text(template: str, record: dict) -> str:
    """LongProc's prompt after the HTML block, filled with the record's public task fields."""
    tail = template.split('{html_str}')[1]
    return tail.lstrip('\n').removeprefix('```').lstrip('\n').format(
        task_topic=record['task_topic'], task_description=record['task_description'],
        filtering_instruction=record['filtering_instruction'], tsv_header=record['tsv_header'])


def build_case(record: dict, root: Path, template: str, split: str) -> tuple[CasePublic, GoldPrivate]:
    page = (root / record['html_path']).read_text(errors='replace')
    public = CasePublic(f"{split}-{record['task_id']}", split, page, (FieldSpec('tsv', task_text(template, record)),))
    return public, GoldPrivate({'tsv': record['gt'].strip() + '\n'}, {'tsv': ()}, {'tsv': ()})
