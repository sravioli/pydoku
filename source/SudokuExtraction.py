# for logging
from loguru import logger
import sys

# SudokuExtraction
import cv2
import numpy as np
from imutils import contours

# import os


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

    def find_contour(
        self, image: np.ndarray, original_image: np.ndarray = None
    ) -> np.ndarray[(1, 2), np.ndarray[(2,), np.intc]]:
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

        if self.debug:
            copy = original_image.copy()
            cv2.drawContours(copy, [grid_contour], 0, (255, 0, 0), 5)
            cv2.imshow("DEBUG: GRID CONTOUR", cv2.resize(copy, (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        logger.debug(f"Return: contour of sudoku grid – {type(grid_contour)}")
        return grid_contour

    def find_corners(
        self,
        grid_contour: np.ndarray[(1, 2), np.ndarray[(2,), np.intc]],
        original_image: np.ndarray = None,
    ) -> list[tuple[int, int]]:
        # the largest contours are the corners of the sudoku
        perimeter = cv2.arcLength(grid_contour, True)
        approx = cv2.approxPolyDP(grid_contour, 0.015 * perimeter, True)
        logger.debug(f"Find approximation of contour – {type(approx)}")

        corners = [(corner[0][0], corner[0][1]) for corner in approx]
        logger.debug(f"Find corners of sudoku grid – {type(corners)}")

        if original_image is not None and self.debug:
            copy = original_image.copy()
            for corner in corners:
                cv2.circle(copy, corner, 10, (255, 0, 255), -1)
            logger.info(f"Draw corners: {self.done}")

            cv2.imshow("DEBUG: GRID CORNERS", cv2.resize(copy, (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if len(corners) == 4:
            # get values of corners
            top_right, top_left, bot_left, bot_right = corners
            logger.debug(f"Swapping corners coordinates order: {self.done}")

            # swap values and return them
            logger.debug(f"Return ordered list of corners – {type(corners)}")
            print(corners)
            return [top_left, top_right, bot_right, bot_left]

    def warp_image(
        self,
        image: np.ndarray,
        corners: list[tuple[int, int]],
    ) -> np.ndarray:
        # to crop the grid the dimension of the latter are needed.
        # Even if the sudoku is a square, calculating the height and width of
        # the grid ensures that any useful part of the image will not be cropped

        # retrieve corners coordinates
        top_left, top_right, bot_right, bot_left = corners
        logger.debug(f"Input corners coordinates – {type(corners)}")

        # get bot values of width
        width_A, width_B = (
            np.sqrt(
                ((bot_right[0] - bot_left[0]) ** 2)
                + ((bot_right[1] - bot_left[1]) ** 2)
            ),
            np.sqrt(
                ((top_right[0] - top_left[0]) ** 2)
                + ((top_right[1] - top_left[1]) ** 2)
            ),
        )
        logger.debug(f"Get bot values of width: {width_A:.3f}, {width_B:.3f}")

        # get bot values of height
        height_A, height_B = (
            np.sqrt(
                ((top_right[0] - bot_right[0]) ** 2)
                + ((top_right[1] - bot_right[1]) ** 2)
            ),
            np.sqrt(
                ((top_left[0] - bot_left[0]) ** 2) + ((top_left[1] - bot_left[1]) ** 2)
            ),
        )
        logger.debug(f"Get both values of height: {height_A:.3f}, {height_B:.3f}")

        # get best width and height
        width, height = (
            max(int(width_A), int(width_B)),
            max(int(height_A), int(height_B)),
        )
        logger.debug(f"Find optimal width and height: {width}, {height}")

        # construct dimensions for the cropped image
        dimensions = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        logger.debug(f"Find dimensions: {type(dimensions)}")

        # convert corners to Numpy array
        corners = np.array(corners, dtype=np.float32)

        # calculate perspective transform and warp prospective
        grid = cv2.getPerspectiveTransform(corners, dimensions)
        logger.debug(f"Calculate perspective transform – {type(grid)}")

        warped_image = cv2.warpPerspective(image, grid, (width, height))
        logger.debug(f"Warp image – {type(warped_image)}")

        if self.debug:
            cv2.imshow("DEBUG: WARPED IMAGE", cv2.resize(warped_image, (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        logger.debug(f"Return warped image – {type(warped_image)}")
        return warped_image

    def extract_cells(self, image: np.ndarray) -> list[np.ndarray]:
        image = original_image.copy()  # don't modify original image
        logger.info(f"Copy original image: {self.done}")

        # turn copy of original image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # VERY IMPORTANT
        logger.debug(f"Convert copy to grayscale: {self.done}")

        # apply adaptive thresholding, then convert to binary image
        image = cv2.bitwise_not(
            cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                101,
                1,
            )
        )
        logger.debug(f"Threshold image and convert to binary: {self.done}")

        # most sudoku are square, but not all. To account for this, find
        # multiple height and width for cells
        edge_height, edge_width = np.shape(image)[0], np.shape(image)[1]
        cell_edge_height, cell_edge_width = edge_height // 9, edge_width // 9
        logger.debug(
            f"Find cell height and width: {cell_edge_height}, {cell_edge_width}"
        )

        # iterate through length&width of the image, extract cells and store
        # them in temporary grid
        temp_grid = []
        for i in range(cell_edge_height, edge_height + 1, cell_edge_height):
            for j in range(cell_edge_width, edge_width + 1, cell_edge_width):

                rows = image[i - cell_edge_height : i]
                temp_grid.append(
                    [rows[k][j - cell_edge_width : j] for k in range(len(rows))]
                )
        logger.debug(f"Create temporary grid of cells – {type(temp_grid)}")

        # Creating an 9x9 array of the images and converting it into a numpy array, so that it is easier to process.

        # create final 9×9 array of images
        final_grid = []
        for i in range(0, len(temp_grid) - 8, 9):
            final_grid.append(temp_grid[i : i + 9])
        logger.debug(f"Create final grid of cells – {type(final_grid)}")

        # convert every item to a Numpy array
        for i in range(9):
            for j in range(9):
                final_grid[i][j] = np.array(final_grid[i][j])

        logger.debug(f"Convert cells to Numpy array – {type(final_grid[0][0])}")

        if self.debug:
            # when debugging, display random cell
            t, k = list(np.random.randint(0, 8, 2))  # get 2 random in [0, 8]

            cv2.imshow("DEBUG: SAMPLE CELL", cv2.resize(final_grid[t][k], (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            logger.debug(f"Show sample image on row {t} and column {k}: {self.done}")

        return final_grid


if __name__ == "__main__":
    import ImageProcessing

    ipr = ImageProcessing.ImageProcessing(debug=True)
    sxt = SudokuExtraction(debug=True)

    # get image and process it
    original_image = ipr.read("/inspo/test_imgs/sudoku.jpg")
    image = ipr.preprocess_image(original_image)

    # get grid contour and grid corners
    grid_contour = sxt.find_contour(image, original_image)  # var2 for debug
    corners = sxt.find_corners(grid_contour, original_image)  # var2 for debug

    # crop+warp image, remove grid lines
    original_image = sxt.warp_image(original_image, corners)
    image = sxt.extract_cells(original_image)
