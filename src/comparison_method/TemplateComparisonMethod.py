import cv2
from cv2.typing import MatLike
from typing_extensions import override

from comparison_method.ImageComparisonMethodBase import ImageComparisonMethodBase
from utils import MAXBYTE, ColorChannel

MASK_SIZE_MULTIPLIER = ColorChannel.Alpha * MAXBYTE * MAXBYTE


class TemplateComparisonMethod(ImageComparisonMethodBase):
    __template_source: MatLike
    __template_mask: MatLike | None

    def __init__(self, source: MatLike):
        super().__init__(source)

        if self.mask is None:
            self.__template_source = self.source
            self.__template_mask = self.mask
        else:
            _, thresh = cv2.threshold(self.mask, 1, 255, cv2.THRESH_BINARY)
            bounding_rect = cv2.boundingRect(thresh)
            x, y, w, h = bounding_rect
            self.__template_source = self.source[y : y + h, x : x + w]
            self.__template_mask = self.mask[y : y + h, x : x + w]

    @override
    def compare(self, capture: MatLike) -> float:
        """
        Compares two images by calculating the L2 Error (square-root of sum of squared error)
        @param source: Image of any given shape
        @param capture: Image matching the dimensions of the source
        @param mask: An image matching the dimensions of the source, but 1 channel grayscale
        @return: The similarity between the images as a number 0 to 1.
        """
        resized_capture = self.resize_capture(capture)
        result = cv2.matchTemplate(
            resized_capture, self.__template_source, cv2.TM_SQDIFF_NORMED, mask=self.__template_mask
        )

        min_val, *_ = cv2.minMaxLoc(result)

        return 1.0 - min_val
