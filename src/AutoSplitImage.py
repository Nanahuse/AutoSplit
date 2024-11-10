import os
import tomllib
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, cast

import cv2
from cv2.typing import MatLike

import error_messages
from comparison_method import (
    ComparisonMethodInterface,
    ImageComparisonMethodBase,
    OcrComparisonMethod,
    OcrSettings,
    get_comparison_method_by_index,
)
from utils import TESSERACT_PATH, imread, is_valid_image

if TYPE_CHECKING:
    from AutoSplit import AutoSplit

START_KEYWORD = "start_auto_splitter"
RESET_KEYWORD = "reset"


class ImageType(IntEnum):
    SPLIT = auto()
    RESET = auto()
    START = auto()


@dataclass
class Rectangle:
    left: int
    right: int
    top: int
    bottom: int

    def is_valid(self) -> bool:
        return (self.right > self.left >= 0) and (self.bottom > self.top >= 0)


class AutoSplitImage:
    image_type: ImageType
    # These values should be overridden by some Defaults if None. Use getters instead
    __delay_time: float | None = None
    __comparison_method_index: int | None = None
    __comparison_method: ComparisonMethodInterface
    __pause_time: float | None = None
    __similarity_threshold: float | None = None
    __rect: Rectangle | None
    __fps_limit: int

    @property
    def is_ocr(self):
        """
        Whether a "split image" is actually for Optical Text Recognition
        based on whether there's any text strings to search for.
        """
        return isinstance(self.__comparison_method, OcrComparisonMethod)

    @property
    def texts(self) -> list[str]:
        if isinstance(self.__comparison_method, OcrComparisonMethod):
            return self.__comparison_method.texts
        return []

    @property
    def byte_array(self) -> MatLike | None:
        if isinstance(self.__comparison_method, ImageComparisonMethodBase):
            return self.__comparison_method.source
        return None

    def get_delay_time(self, default: "AutoSplit | int"):
        """Get image's delay time or fallback to the default value from spinbox."""
        if self.__delay_time is not None:
            return self.__delay_time
        if isinstance(default, int):
            return default
        return default.settings_dict["default_delay_time"]

    def get_comparison_method_index(self, default: "AutoSplit | int"):
        """Get image's comparison or fallback to the default value from combobox."""
        if self.__comparison_method_index is not None:
            return self.__comparison_method_index
        if isinstance(default, int):
            return default
        return default.settings_dict["default_comparison_method"]

    def get_pause_time(self, default: "AutoSplit | float"):
        """Get image's pause time or fallback to the default value from spinbox."""
        if self.__pause_time is not None:
            return self.__pause_time
        if isinstance(default, (float, int)):
            return default
        return default.settings_dict["default_pause_time"]

    def get_similarity_threshold(self, default: "AutoSplit | float"):
        """Get image's similarity threshold or fallback to the default value from spinbox."""
        if self.__similarity_threshold is not None:
            return self.__similarity_threshold
        if isinstance(default, (float, int)):
            return default
        return default.settings_dict["default_similarity_threshold"]

    def get_fps_limit(self, default: "AutoSplit") -> int:
        """Get image's fps limit or fallback to the default value from spinbox."""
        if self.__fps_limit != 0:
            return self.__fps_limit
        return default.settings_dict["fps_limit"]

    def __init__(self, path: str):
        self.path = path
        self.filename = os.path.split(path)[-1].lower()
        self.flags = flags_from_filename(self.filename)
        self.loops = loop_from_filename(self.filename)
        self.__delay_time = delay_time_from_filename(self.filename)
        self.__comparison_method_index = comparison_method_from_filename(self.filename)
        self.__pause_time = pause_from_filename(self.filename)
        self.__similarity_threshold = threshold_from_filename(self.filename)
        self.__comparison_method = ComparisonMethodInterface()
        self.__rect = None
        self.__fps_limit = 0

        if path.endswith("txt"):
            self.__parse_text_file(path)
        else:
            self.__read_image_bytes(path)

        if START_KEYWORD in self.filename:
            self.image_type = ImageType.START
        elif RESET_KEYWORD in self.filename:
            self.image_type = ImageType.RESET
        else:
            self.image_type = ImageType.SPLIT

    def __parse_text_file(self, path: str):
        if not TESSERACT_PATH:
            error_messages.tesseract_missing(path)
            return

        with open(path, mode="rb") as f:
            data = cast(OcrSettings, tomllib.load(f))

        rect = Rectangle(data["left"], data["right"], data["top"], data["bottom"])
        fps_limit = data.get("fps_limit", 0)

        texts = [text.lower().strip() for text in data["texts"]]
        ocr_comparison_methods = data.get("methods", [0])

        # Check for invalid negative values
        if all(value >= 0 for value in [*ocr_comparison_methods, fps_limit]) and rect.is_valid():
            error_messages.wrong_ocr_values(path)
            return

        self.__comparison_method = OcrComparisonMethod(texts, ocr_comparison_methods)
        self.__rect = rect
        self.__fps_limit = fps_limit

    def __read_image_bytes(self, path: str):
        image = imread(path, cv2.IMREAD_UNCHANGED)
        if not is_valid_image(image):
            self.__comparison_method = ComparisonMethodInterface()
            error_messages.image_type(path)
            return

        comparison_method = get_comparison_method_by_index(self.__comparison_method_index)
        self.__comparison_method = comparison_method(image)

    def check_flag(self, flag: int):
        return self.flags & flag == flag

    def compare_with_capture(self, default: "AutoSplit | int", capture: MatLike | None):
        """
        Compare image with capture using image's comparison method. Falls back to combobox.

        For OCR text files:
            extract image text from rectangle position and compare it with the expected string.
        """
        if not is_valid_image(capture):
            return 0.0

        if (self.__comparison_method_index is None) and (
            isinstance(self.__comparison_method, ImageComparisonMethodBase)
        ):
            comparison_method_index = self.get_comparison_method_index(default)
            comparison_method = get_comparison_method_by_index(comparison_method_index)
            if type(self.__comparison_method) is not comparison_method:
                self.__comparison_method = comparison_method(self.__comparison_method.source)

        return self.__comparison_method.compare(
            capture
            if self.__rect is None
            else capture[self.__rect.top : self.__rect.bottom, self.__rect.left : self.__rect.right]
        )


if True:
    from split_parser import (
        comparison_method_from_filename,
        delay_time_from_filename,
        flags_from_filename,
        loop_from_filename,
        pause_from_filename,
        threshold_from_filename,
    )
