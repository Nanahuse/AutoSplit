import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray
from scipy import fft
from typing_extensions import override

from comparison_method.ImageComparisonMethodBase import ImageComparisonMethodBase
from utils import is_valid_image


class PHashComparisonMethod(ImageComparisonMethodBase):
    __source_hash: NDArray[np.bool_]

    def __init__(self, source: MatLike):
        super().__init__(source)

        self.__source_hash = self.__cv2_phash(self.source)

    @override
    def compare(self, capture: MatLike) -> float:
        """
        Compares the Perceptual Hash of the two given images and returns the similarity between the two.

        @param capture: Image of any given shape as a numpy array
        @return: The similarity between the hashes of the image as a number 0 to 1.
        """  # noqa: E501
        resized_capture = self.resize_capture(capture)
        capture_hash = self.__cv2_phash(resized_capture)
        hash_diff = np.count_nonzero(self.__source_hash != capture_hash)

        return 1.0 - (hash_diff / 64.0)

    def __cv2_phash(
        self, image: MatLike, hash_size: int = 8, highfreq_factor: int = 4
    ) -> NDArray[np.bool_]:
        """Implementation copied from https://github.com/JohannesBuchner/imagehash/blob/38005924fe9be17cfed145bbc6d83b09ef8be025/imagehash/__init__.py#L260 ."""  # noqa: E501
        # OpenCV has its own pHash comparison implementation in `cv2.img_hash`,
        # but it requires contrib/extra modules and is inaccurate
        # unless we precompute the size with a specific interpolation.
        # See: https://github.com/opencv/opencv_contrib/issues/3295#issuecomment-1172878684
        #
        # pHash = cv2.img_hash.PHash.create()
        # source = cv2.resize(source, (8, 8), interpolation=cv2.INTER_AREA)
        # capture = cv2.resize(capture, (8, 8), interpolation=cv2.INTER_AREA)
        # source_hash = pHash.compute(source)
        # capture_hash = pHash.compute(capture)
        # hash_diff = pHash.compare(source_hash, capture_hash)

        # Apply the mask to the image before calculating the pHash.
        # As a result of this, this function
        # is not going to be very helpful for large masks as the images
        # when shrinked down to 8x8 will mostly be the same.
        if is_valid_image(self.mask):
            image = cv2.bitwise_and(image, image, mask=self.mask)

        img_size = hash_size * highfreq_factor
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
        dct = fft.dct(fft.dct(image, axis=0), axis=1)
        dct_low_frequency = dct[:hash_size, :hash_size]
        median = np.median(dct_low_frequency)
        return dct_low_frequency > median
