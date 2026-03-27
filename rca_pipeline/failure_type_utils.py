"""
Helpers for splitting RCA predictions by failure type.
"""

from typing import Dict, List, Sequence, Tuple


KNOWN_FAILURE_TYPES = ("hard", "gray", "cascade")


def split_logits_by_failure_type(
    data_list: Sequence[object],
    y_true: Sequence[int],
    y_logits: Sequence[object],
) -> Dict[str, Dict[str, List[object]]]:
    """
    Split labels/logits by failure type.

    Returns a dict:
        {
          "hard": {"y_true": [...], "y_logits": [...]},
          "gray": {"y_true": [...], "y_logits": [...]},
          "cascade": {"y_true": [...], "y_logits": [...]},
          "other": {"y_true": [...], "y_logits": [...]},
        }
    """
    if not (len(data_list) == len(y_true) == len(y_logits)):
        raise ValueError("data_list, y_true, and y_logits must have same length")

    out: Dict[str, Dict[str, List[object]]] = {
        "hard": {"y_true": [], "y_logits": []},
        "gray": {"y_true": [], "y_logits": []},
        "cascade": {"y_true": [], "y_logits": []},
        "other": {"y_true": [], "y_logits": []},
    }

    for d, yi, lo in zip(data_list, y_true, y_logits):
        ft = getattr(d, "failure_type", "other")
        bucket = ft if ft in KNOWN_FAILURE_TYPES else "other"
        out[bucket]["y_true"].append(yi)
        out[bucket]["y_logits"].append(lo)

    return out
