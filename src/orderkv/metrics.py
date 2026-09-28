"""LongProc HTML-to-TSV scoring through the official evaluator."""
import contextlib
import io

from orderkv.longproc_official import eval_html_to_tsv


def row_f1(prediction, gold):
    """LongProc's official HTML-to-TSV row precision, recall and F1."""
    with contextlib.redirect_stdout(io.StringIO()):  # the evaluator prints every malformed row
        metrics, _ = eval_html_to_tsv(prediction, {'reference_output': gold})
    return {key: float(metrics[key]) for key in ('precision', 'recall', 'f1')}
