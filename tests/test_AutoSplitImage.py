# ruff: noqa: S101
# pyright: reportUnknownMemberType=false

from pathlib import Path
from unittest.mock import patch

import cv2
import pytest

from AutoSplitImage import AutoSplitImage
from comparison_method import ComparisonMethod
from utils import imread


def image_path_str(filename: str):
    test_doc_dir = Path(__file__).parent / "docs"
    return str(test_doc_dir / filename)


def comparison_method_allow_transparency():
    return [method for method in ComparisonMethod if method is not ComparisonMethod.PHash]


def mock_imread(filename: str, flags: int):  # noqa: ARG001
    return imread(image_path_str("00_test_image.png"), flags)


@patch("AutoSplitImage.imread", mock_imread)
def test_get_comparison_method_index():
    image = AutoSplitImage("00.png")
    assert image.get_comparison_method_index(0) == ComparisonMethod.L2Norm
    assert image.get_comparison_method_index(1) == ComparisonMethod.Histograms
    assert image.get_comparison_method_index(2) == ComparisonMethod.PHash

    image = AutoSplitImage("00_^1^.png")
    assert image.get_comparison_method_index(0) == ComparisonMethod.Histograms
    assert image.get_comparison_method_index(1) == ComparisonMethod.Histograms
    assert image.get_comparison_method_index(2) == ComparisonMethod.Histograms


def test_byte_array():
    image = AutoSplitImage(image_path_str("00_test_image.png"))
    assert image.byte_array is not None
    assert image.byte_array.shape == (240, 320, 4)

    image = AutoSplitImage(image_path_str("00_test_image_transparency_compress.png"))
    assert image.byte_array is not None
    assert image.byte_array.shape == (240, 320, 4)

    image = AutoSplitImage(image_path_str("00_test_image_transparency_uncompress.png"))
    assert image.byte_array is not None
    assert image.byte_array.shape == (300, 400, 4)


@pytest.mark.parametrize(("comparison_method_indexes"), ComparisonMethod)
def test_compare_with_capture_same_data(comparison_method_indexes: int):
    image = AutoSplitImage(image_path_str("00_test_image.png"))
    capture = imread(image_path_str("00_test_image.png"))
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2BGRA)
    assert image.compare_with_capture(comparison_method_indexes, capture) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("comparison_method_indexes", "expected"),
    zip(ComparisonMethod, [0.73444853, 0.69336759, 0.96875], strict=True),
)
def test_compare_with_capture_diff_data(comparison_method_indexes: int, expected: float):
    image = AutoSplitImage(image_path_str("00_test_image.png"))
    capture = imread(image_path_str("01_test_image.png"))
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2BGRA)
    assert image.compare_with_capture(comparison_method_indexes, capture) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("comparison_method_indexes", "expected"),
    zip(
        comparison_method_allow_transparency(),
        [0.68961008, 0.69326769],
        strict=True,
    ),
)
def test_compare_with_capture_same_data_transparency_compress(
    comparison_method_indexes: int, expected: float
):
    image = AutoSplitImage(image_path_str("00_test_image_transparency_compress.png"))
    capture = imread(image_path_str("01_test_image.png"))
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2BGRA)
    assert image.compare_with_capture(comparison_method_indexes, capture) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("comparison_method_indexes", "expected"),
    zip(
        comparison_method_allow_transparency(),
        [0.42887980, 0.4288797],
        strict=True,
    ),
)
def test_compare_with_capture_same_data_transparency_uncompress(
    comparison_method_indexes: int, expected: float
):
    image = AutoSplitImage(image_path_str("00_test_image_transparency_uncompress.png"))
    capture = imread(image_path_str("01_test_image.png"))
    capture = cv2.cvtColor(capture, cv2.COLOR_BGR2BGRA)
    assert image.compare_with_capture(comparison_method_indexes, capture) == pytest.approx(expected)
