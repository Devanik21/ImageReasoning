"""Evaluation metrics."""

def compute_accuracy(preds, labels):
    if not preds:
        return 0.0
    if len(preds) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    return sum(p == l for p, l in zip(preds, labels)) / len(preds)
