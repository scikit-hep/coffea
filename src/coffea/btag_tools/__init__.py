"""BTag tools: CMS analysis-level b-tagging corrections and uncertainties

These classes provide computation of CMS b-tagging and mistagging
corrections and uncertainties on columnar data.
"""

from .btagscalefactor import BTagScaleFactor

__all__ = [
    "BTagScaleFactor",
]
