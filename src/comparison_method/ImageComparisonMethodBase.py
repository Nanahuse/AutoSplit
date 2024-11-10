from math import sqrt

import cv2
import numpy as np
from cv2.typing import MatLike

from comparison_method.ComparisonMethodInterface import ComparisonMethodInterface
from utils import BGR_CHANNEL_COUNT, BGRA_CHANNEL_COUNT, MAXBYTE, ColorChannel, ImageShape

# Resize to these width and height so that FPS performance increases
COMPARISON_RESIZE_WIDTH = 320
COMPARISON_RESIZE_HEIGHT = 240
COMPARISON_RESIZE = (COMPARISON_RESIZE_WIDTH, COMPARISON_RESIZE_HEIGHT)
COMPARISON_RESIZE_AREA = COMPARISON_RESIZE_WIDTH * COMPARISON_RESIZE_HEIGHT
MASK_LOWER_BOUND = np.array([0, 0, 0, 1], dtype=np.uint8)
MASK_UPPER_BOUND = np.array([MAXBYTE, MAXBYTE, MAXBYTE, MAXBYTE], dtype=np.uint8)


class ImageComparisonMethodBase(ComparisonMethodInterface):
    source: MatLike
    mask: MatLike | None

    def __init__(self, image: MatLike):
        # If image has transparency, create a mask
        if self.__check_if_image_has_transparency(image):
            # Adaptively determine the target size according to
            # the number of nonzero elements in the alpha channel of the split image.
            # This may result in images bigger than COMPARISON_RESIZE if there's plenty of transparency. # noqa: E501
            # Which wouldn't incur any performance loss in methods where masked regions are ignored.
            scale = min(
                1.0,
                sqrt(COMPARISON_RESIZE_AREA / cv2.countNonZero(image[:, :, ColorChannel.Alpha])),
            )

            image = cv2.resize(
                image,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

            # Mask based on adaptively resized, nearest neighbor interpolated split image
            mask = cv2.inRange(image, MASK_LOWER_BOUND, MASK_UPPER_BOUND)
        else:
            image = cv2.resize(image, COMPARISON_RESIZE, interpolation=cv2.INTER_NEAREST)
            mask = None

            # Add Alpha channel if missing
            if image.shape[ImageShape.Channels] == BGR_CHANNEL_COUNT:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        self.source = image
        self.mask = mask

    def resize_capture(self, capture: MatLike):
        """
        Resize given image to same shape as source image.

        @param capture: Given image
        @return: Resized image.
        """
        if capture.shape[1::-1] == self.source.shape[1::-1]:
            return capture
        return cv2.resize(capture, self.source.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def __check_if_image_has_transparency(image: MatLike):
        # Check if there's a transparency channel (4th channel)
        # and if at least one pixel is transparent (< 255)
        if image.shape[ImageShape.Channels] != BGRA_CHANNEL_COUNT:
            return False
        mean: float = image[:, :, ColorChannel.Alpha].mean()
        if mean == 0:
            # Non-transparent images code path is usually faster and simpler, so let's return that
            return False
            # TODO: error message if all pixels are transparent
            # (the image appears as all black in windows,
            # so it's not obvious for the user what they did wrong)

        return mean != MAXBYTE
