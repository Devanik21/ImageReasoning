"""Evaluation metrics."""

def compute_accuracy(preds, labels):
    return sum(p == l for p, l in zip(preds, labels)) / len(preds)
