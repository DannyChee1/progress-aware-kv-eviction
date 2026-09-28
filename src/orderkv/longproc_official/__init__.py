"""LongProc's official HTML-to-TSV evaluator, vendored unchanged from
https://github.com/princeton-pli/LongProc at commit 673ec4c230876e941d674116d66d28783c186b15 (Apache-2.0).

html_to_tsv_evaluator.py is a verbatim copy; longproc_data.py holds only eval_html_to_tsv.
"""
from .longproc_data import eval_html_to_tsv

__all__ = ['eval_html_to_tsv']
