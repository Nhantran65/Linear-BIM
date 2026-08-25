"""Read-only metrics over frozen predictions."""

from .frozen import evaluate_frozen, metric_bundle

__all__ = ["evaluate_frozen", "metric_bundle"]
