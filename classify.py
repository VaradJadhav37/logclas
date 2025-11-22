# classify.py  (lazy-import wrapper)
from typing import Iterable, List, Tuple
import traceback

def classify(inputs: Iterable[Tuple[str,str]]) -> List[str]:
    """
    inputs: iterable of (source, log_message)
    returns: list of labels (str), length == len(inputs)

    Lazy-loads processor modules so importing classify does not trigger heavy imports.
    """
    # Lazy imports
    classify_with_regex = None
    classify_with_bert = None
    classify_with_llm = None

    try:
        from processor_regex import classify_with_regex as _r
        classify_with_regex = _r
    except Exception as e:
        print("Warning: failed to import processor_regex:", e)
        print(traceback.format_exc())

    try:
        from processor_bert import classify_with_bert as _b
        classify_with_bert = _b
    except Exception as e:
        print("Warning: failed to import processor_bert:", e)
        print(traceback.format_exc())

    try:
        from processor_llm import classify_with_llm as _l
        classify_with_llm = _l
    except Exception as e:
        print("Warning: failed to import processor_llm:", e)
        print(traceback.format_exc())

    labels: List[str] = []
    for src, msg in inputs:
        label = None

        # 1) regex
        if classify_with_regex is not None:
            try:
                label = classify_with_regex(str(msg))
            except Exception as e:
                print("regex error:", e)

        # 2) bert
        if (label is None or label == "") and classify_with_bert is not None:
            try:
                label = classify_with_bert(str(msg))
            except Exception as e:
                print("bert error:", e)

        # 3) llm fallback
        if (label is None or label == "Unclassified") and classify_with_llm is not None:
            try:
                label = classify_with_llm(str(msg))
            except Exception as e:
                print("llm error:", e)

        if label is None:
            label = "Unclassified"
        labels.append(label)

    return labels
