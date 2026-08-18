"""Standalone Linear-BIM research package.

Historical experiment directories remain provenance. New code is self-contained
and does not import those directories or depend on the original workspace.
"""

from .models.bim import BIM, CurrentOnly

__all__ = ["BIM", "CurrentOnly"]
__version__ = "0.1.0"
