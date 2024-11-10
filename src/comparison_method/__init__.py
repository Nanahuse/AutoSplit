from enum import IntEnum

from .ComparisonMethodInterface import ComparisonMethodInterface
from .HistogramComparisonMethod import HistogramComparisonMethod
from .ImageComparisonMethodBase import ImageComparisonMethodBase
from .L2NormComparisonMethod import L2NormComparisonMethod
from .OcrComparisonMethod import OcrComparisonMethod, OcrSettings
from .PHashComparisonMethod import PHashComparisonMethod


class ComparisonMethod(IntEnum):
    L2Norm = 0
    Histograms = 1
    PHash = 2


def get_comparison_method_by_index(index: int | None) -> type[ImageComparisonMethodBase]:
    """Get the comparison method from index."""
    match index:
        case ComparisonMethod.L2Norm:
            return L2NormComparisonMethod
        case ComparisonMethod.Histograms:
            return HistogramComparisonMethod
        case ComparisonMethod.PHash:
            return PHashComparisonMethod
        case _:
            return ImageComparisonMethodBase


__all__ = [
    "ComparisonMethod",
    "ComparisonMethodInterface",
    "HistogramComparisonMethod",
    "ImageComparisonMethodBase",
    "L2NormComparisonMethod",
    "OcrComparisonMethod",
    "OcrSettings",
    "PHashComparisonMethod",
    "get_comparison_method_by_index",
]
