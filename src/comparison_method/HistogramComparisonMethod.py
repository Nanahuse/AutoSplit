import cv2
from cv2.typing import MatLike
from typing_extensions import override

from comparison_method.ImageComparisonMethodBase import ImageComparisonMethodBase
from utils import MAXBYTE, ColorChannel

MAXRANGE = MAXBYTE + 1
CHANNELS = (ColorChannel.Red.value, ColorChannel.Green.value, ColorChannel.Blue.value)
HISTOGRAM_SIZE = (8, 8, 8)
RANGES = (0, MAXRANGE, 0, MAXRANGE, 0, MAXRANGE)


class HistogramComparisonMethod(ImageComparisonMethodBase):
    __source_hist: MatLike

    def __init__(self, source: MatLike):
        super().__init__(source)

        source_hist = cv2.calcHist([self.source], CHANNELS, self.mask, HISTOGRAM_SIZE, RANGES)
        cv2.normalize(source_hist, source_hist)
        self.__source_hist = source_hist

    @override
    def compare(self, capture: MatLike) -> float:
        """
        Compares two images by calculating their histograms, normalizing
        them, and then comparing them using Bhattacharyya distance.

        @param capture: An image matching the shape, dimensions and format of the source
        @return: The similarity between the histograms as a number 0 to 1.
        """
        resized_capture = self.resize_capture(capture)
        capture_hist = cv2.calcHist([resized_capture], CHANNELS, self.mask, HISTOGRAM_SIZE, RANGES)

        cv2.normalize(capture_hist, capture_hist)

        return 1.0 - cv2.compareHist(self.__source_hist, capture_hist, cv2.HISTCMP_BHATTACHARYYA)
