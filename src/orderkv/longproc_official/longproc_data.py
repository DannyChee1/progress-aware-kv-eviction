# Copied from princeton-pli/LongProc longproc/longproc_data.py at commit 673ec4c2 (Apache-2.0, see LICENSE).
# Only eval_html_to_tsv and the imports it needs; the function body is unchanged.
import re

from .html_to_tsv_evaluator import evaluate_html_to_csv_compute_metrics


def eval_html_to_tsv(prediction: str, example: dict):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    # metric: f1, precision, recall, extraction_rate
    try:
        # prediction should be wrapped with ```tsv and ``` to be parsed
        search = re.search(r'```tsv([\s\S]*)```', prediction)
        # exactly one group should be matched
        if search is not None:
            prediction = search.group(1).strip()
        else:
            if "```tsv" in prediction:
                prediction = prediction.split("```tsv")[1]
            prediction = prediction.strip() # use all
    except Exception as e:
        ## if "TSV:" is not in the output, then return 0.0 for all metrics because the model didn't follow format
        return {"f1": .0, "precision": .0, "recall": .0,"extraction_rate": .0}, {"parsed_output": None,"error_msg": str(e)}
    eval_results = evaluate_html_to_csv_compute_metrics(prediction, example["reference_output"])
    return {"f1": eval_results["f1"], "precision": eval_results["precision"], "recall": eval_results["recall"],"extraction_rate": 1.0 if eval_results["error"] is None else 0.0}, {"parsed_output": prediction,"error_msg": eval_results["error"]}
