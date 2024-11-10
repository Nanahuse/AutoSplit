from math import sqrt

import cv2
from cv2.typing import MatLike
from typing_extensions import override

from comparison_method.ImageComparisonMethodBase import ImageComparisonMethodBase
from utils import MAXBYTE, ColorChannel, is_valid_image

MASK_SIZE_MULTIPLIER = ColorChannel.Alpha * MAXBYTE * MAXBYTE


class L2NormComparisonMethod(ImageComparisonMethodBase):
    __max_error: float

    def __init__(self, source: MatLike):
        super().__init__(source)

        self.__max_error = (
            sqrt(self.source.size) * MAXBYTE
            if not is_valid_image(self.mask)
            else sqrt(cv2.countNonZero(self.mask) * MASK_SIZE_MULTIPLIER)
        )

    @override
    def compare(self, capture: MatLike) -> float:
        """
        Compares two images by calculating the L2 Error (square-root of sum of squared error).

        @param capture: Image matching the dimensions of the source
        @return: The similarity between the images as a number 0 to 1.
        """
        resized_capture = self.resize_capture(capture)

        error = cv2.norm(self.source, resized_capture, cv2.NORM_L2, self.mask)

        if not self.__max_error:
            return 0.0
        return 1.0 - (error / self.__max_error)
