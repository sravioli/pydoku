# for logging
from loguru import logger
import sys

# SudokuExtraction
import cv2
import numpy as np

from rdp import rdp


logger.remove()
logger_format = (
    "<green>{time:DD/MM/YYYY – HH:mm:ss}</green> "
    + "| <lvl>{level: <8}</lvl> "
    + "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    + "– <lvl>{message}</lvl>"
)
logger.add(
    sys.stdout,
    colorize=True,
    format=logger_format,
)


class SudokuExtraction:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.done = "\033[92m✓\033[39m"

    def find_contours(
        self, image: np.ndarray, original_image: np.ndarray = None
    ) -> np.ndarray:
        # use Canny Edge Detection Algorithm to detect edges
        image = cv2.Canny(image, 50, 150)
        logger.info(f"Apply Canny edge detection algorithm: {self.done}")

        # a contour is basically any closed shape. Find contours and hierarchy
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        logger.debug(f"Number of found contours: {len(contours)}")

        # find largest contour. Assuming that the largest contour is the sudoku
        # grid. A contour is basically a set of coordinates
        if len(contours) != 0:
            grid_contour = max(contours, key=cv2.contourArea)
            logger.debug(f"Find grid contour – length: {len(grid_contour)}")

            cv2.drawContours(original_image, [grid_contour], 0, (255, 0, 0), 5)

        if self.debug:
            cv2.imshow("DEBUG: GRID CONTOUR", cv2.resize(original_image, (320, 320)))
            cv2.waitKey(1500)
            cv2.destroyAllWindows()

        logger.debug(f"Return: contour of sudoku grid – {type(grid_contour)}")
        return grid_contour

    def find_corners(
        self, grid_contour: np.ndarray, original_image: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        # the largest contours are the corners of the sudoku
        pass


if __name__ == "__main__":
    import ImageProcessing

    ipr = ImageProcessing.ImageProcessing(debug=True)
    sxt = SudokuExtraction(debug=True)

    # get image and process it
    original_image = ipr.read("/inspo/test_imgs/sudoku.jpg")
    image = ipr.preprocess_image(original_image)

    # get contours
    contours = sxt.find_contours(image, original_image)
    # corners = sxt.find_corners(image, contours)
