from collections.abc import Callable
from typing import NotRequired, TypedDict

import cv2
import Levenshtein
import numpy as np
from cv2.typing import MatLike
from typing_extensions import override

from comparison_method.ComparisonMethodInterface import MAX_VALUE, ComparisonMethodInterface
from utils import run_tesseract


class OcrSettings(TypedDict):
    texts: list[str]
    left: int
    right: int
    top: int
    bottom: int
    methods: NotRequired[list[int]]
    fps_limit: NotRequired[int]


def compare_submatch(a: str, b: str):
    return float(a in b)


def compare_dummy(*_: object):
    return 0.0


class OcrComparisonMethod(ComparisonMethodInterface):
    __texts: list[str]
    __ocr_comparison_methods: list[Callable[[str, str], float]]

    def __init__(self, texts: list[str], ocr_comparison_methods: list[int]):
        self.__texts = texts
        self.__ocr_comparison_methods = [
            self.__get_ocr_comparison_method_by_index(i) for i in ocr_comparison_methods
        ]

    @property
    def texts(self) -> list[str]:
        return self.__texts

    @override
    def compare(self, capture: MatLike) -> float:
        """
        Compares the extracted text of the given image and returns the similarity between the two texts.
        The best match of all texts and methods is returned.

        @param capture: Image of any given shape as a numpy array
        @return: The similarity between the text in the image and the text supplied as a number 0 to 1.
        """  # noqa: E501
        png = np.array(cv2.imencode(".png", capture)[1]).tobytes()
        # Especially with stylised characters, OCR could conceivably get the right
        # letter, but mix up the casing (m/M, o/O, t/T, etc.)
        image_string = run_tesseract(png).lower().strip()

        ratio = 0.0
        for text in self.__texts:
            for method in self.__ocr_comparison_methods:
                ratio = max(ratio, method(text, image_string))
                if ratio == MAX_VALUE:
                    return ratio  # we found the best match; try to return early
        return ratio

    @staticmethod
    def __get_ocr_comparison_method_by_index(
        comparison_method_index: int,
    ) -> Callable[[str, str], float]:
        match comparison_method_index:
            case 0:
                return Levenshtein.ratio
            case 1:
                return compare_submatch
            case _:
                return compare_dummy
