from typing import List, Dict, Tuple

def extract_tuples(sub_queries: List[Dict]) -> set[Tuple[str, int, str]]:
    """Extracts immutable routing tuples for set-math evaluation."""
    return {(sq.get("ticker"), sq.get("fiscal_year"), sq.get("item_number")) for sq in sub_queries}

def grade_decomposition(expected: List[Dict], actual: List[Dict]) -> Dict[str, float]:
    """Deterministically grades Planner and Router accuracy."""
    expected_set = extract_tuples(expected)
    actual_set = extract_tuples(actual)
    
    # Handle Adversarial/Empty cases
    if not expected_set:
        return {
            "recall": 1.0 if not actual_set else 0.0,
            "precision": 1.0 if not actual_set else 0.0,
            "routing_accuracy": 1.0
        }

    # Planner Math
    true_positives = expected_set.intersection(actual_set)
    recall = len(true_positives) / len(expected_set) if expected_set else 0.0
    precision = len(true_positives) / len(actual_set) if actual_set else 0.0

    # Router Math (Only grade routing for true positive tuples)
    correct_routes = 0
    for exp_sq in expected:
        exp_tuple = (exp_sq.get("ticker"), exp_sq.get("fiscal_year"), exp_sq.get("item_number"))
        if exp_tuple in true_positives:
            # Find matching actual sub-query
            act_sq = next(sq for sq in actual if (sq.get("ticker"), sq.get("fiscal_year"), sq.get("item_number")) == exp_tuple)
            if exp_sq.get("retrieval_mode") == act_sq.get("retrieval_mode"):
                correct_routes += 1
                
    routing_accuracy = correct_routes / len(true_positives) if true_positives else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "routing_accuracy": routing_accuracy
    }