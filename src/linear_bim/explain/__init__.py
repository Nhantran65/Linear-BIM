"""BIM-Explain attribution APIs.

``core`` is the 14-coordinate attribution and ``full_core`` is the distinct
8-group local/history attribution.  They intentionally are not aliases.
"""

from .core import CLASS_NAMES as COORDINATE_CLASS_NAMES
from .full_core import PURE_GROUP_NAMES
from .frozen import coordinate_explain, frozen_coordinate_explain, history_explain

__all__ = [
    "COORDINATE_CLASS_NAMES",
    "PURE_GROUP_NAMES",
    "coordinate_explain",
    "frozen_coordinate_explain",
    "history_explain",
]
