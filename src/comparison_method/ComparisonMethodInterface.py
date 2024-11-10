from cv2.typing import MatLike

MAX_VALUE = 1.0


class ComparisonMethodInterface:
    def compare(self, capture: MatLike) -> float:  # noqa: PLR6301
        """
        Compare capture using image's comparison method.

        @param capture: An image matching the shape, dimensions and format of the source
        @return: The similarity as a number 0 to 1.
        """
        return 0.0
