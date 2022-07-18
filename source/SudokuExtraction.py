# for logging
from loguru import logger
import sys

# SudokuExtraction
import cv2
import numpy as np

# import operator


logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:DD/MM/YYYY – HH:mm:ss}</green> | <lvl>{level: <8}</lvl> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> – <lvl>{message}</lvl>",
)


class SudokuExtraction:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.done = "\033[92m✓\033[39m"

    def find_contours(
        self, image: np.ndarray, original_image: np.ndarray = None
    ) -> tuple[np.ndarray, ...]:
        # use Canny Edge Detection Algorithm to detect edges
        image = cv2.Canny(image, 50, 150)
        logger.info(f"Apply Canny edge detection algorithm: {self.done}")

        if self.debug:
            logger.debug("Showing resulting image from canny edge detection")
            cv2.imshow("DEBUG: CANNY", cv2.resize(image, (320, 320)))
            cv2.waitKey(1500)
            cv2.destroyAllWindows()

        # a contour is basically any closed shape
        # find contours and hierarchy (not used)
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        logger.debug(f"Number of found contours: {len(contours)}")

        if original_image is not None and self.debug:
            cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3)
            logger.debug(f"Draw contours on original image: {self.done}")

            cv2.imshow("DEBUG: CONTOUR (ALL)", cv2.resize(original_image, (320, 320)))
            cv2.waitKey(1500)
            cv2.destroyAllWindows()

        # assuming sudoku has the largest contour area, find it
        if len(contours) != 0:
            grid_contour = max(contours, key=cv2.contourArea)
            logger.info(f"Largest contour size: {len(grid_contour)}")

            bot_left = tuple(grid_contour[grid_contour[:, :, 0].argmin()][0])
            bot_right = tuple(grid_contour[grid_contour[:, :, 0].argmax()][0])
            top_right = tuple(grid_contour[grid_contour[:, :, 1].argmin()][0])
            bottom = tuple(grid_contour[grid_contour[:, :, 1].argmax()][0])

            # Draw dots onto image
            cv2.drawContours(original_image, [grid_contour], -1, (36, 255, 12), 2)
            cv2.circle(original_image, bot_left, 8, (0, 50, 255), -1)  # red
            cv2.circle(original_image, bot_right, 8, (0, 255, 255), -1)  # yellow
            cv2.circle(original_image, top_right, 8, (255, 50, 0), -1)  # blue
            cv2.circle(original_image, bottom, 8, (255, 255, 0), -1)  # cyan

            cv2.imshow("DEBUG: grid", cv2.resize(original_image, (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # # find coords of top left corner + width & height
            # x, y, w, h = cv2.boundingRect(grid_contour)
            # logger.debug(f"Top left coords: ({x}, {y}) – Width, Height: {w}, {h}")

        # if original_image is not None and self.debug:
        #     # draw rectangle over grid
        #     cv2.rectangle(original_image, (x, y), (x + w, y + h), (255, 0, 0), 5)

        #     # draw circle on found corners
        #     cv2.circle(original_image, (x, y), 10, (0, 0, 255), 5)
        #     cv2.circle(original_image, (x + w, y + h), 10, (0, 0, 255), 5)

        #     cv2.imshow("DEBUG: GRID", cv2.resize(original_image, (320, 320)))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        return contours

    def find_corners(self, image, approx):
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
