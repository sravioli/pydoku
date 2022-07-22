# for logging
from loguru import logger
import sys

# SudokuExtractor
import cv2
import numpy as np

from scipy import ndimage
from keras.models import Sequential


# logger.remove()
logger_format = (
    "<green>{time:DD/MM/YYYY – HH:mm:ss}</green> "
    + "| <lvl>{level: <8}</lvl> "
    + "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    + "– <lvl>{message}</lvl>"
)
logger.add(sys.stdout, colorize=True, format=logger_format)


class SudokuExtractor:
    def __init__(self, debug: bool = False):
        """Utility class to extract the sudoku grid from an image.

        Args:
            debug (bool, optional): Whether or not to display the various
                steps. Defaults to False.
        """
        self.debug = debug
        self.done = "\033[92m✓\033[39m"

    def find_contour(
        self, image: np.ndarray, original_image: np.ndarray = None
    ) -> np.ndarray[(1, 2), np.ndarray[(2,), np.intc]]:
        """Finds the sudoku grid contour by assuming that the latter is the largest contour in the image.

        Args:
            image (np.ndarray): The processed image from which to extract the
                contour.
            original_image (np.ndarray, optional): The original image. Used for
                debugging purposes. Defaults to None.

        Returns:
            np.ndarray: The contour of the sudoku grid.
        """
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
        """Finds the corners of the sudoku grid from the given contour.

        Args:
            grid_contour (np.ndarray): The contour of the sudoku grid.
            original_image (np.ndarray, optional): The original image, used for debugging purposes. Defaults to None.

        Returns:
            list[tuple[int, int]]: A list containing 4 tuples containing 2
                ints, for example: [(<int>, <int>) * 4]. The order of the
                coordinates is as follows: [top_left, top_right, bot_right,
                bot_left].
        """
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
            return [top_left, top_right, bot_right, bot_left]

    def warp_image(
        self,
        image: np.ndarray,
        corners: list[tuple[int, int]],
    ) -> np.ndarray:
        """Warps the given image by using the given corners.

        Args:
            image (np.ndarray): The (original) image to warp.
            corners (list[tuple[int, int]]): A list containing 4 tuples
                containing 2 ints, for example: [(<int>, <int>) * 4]. The
                preferred order of the coordinates is as follows: [top_left,
                top_right, bot_right, bot_left].

        Returns:
            np.ndarray: The warped image.
        """
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

    def remove_grid(self, image):

        ker = image.shape[0] // 9
        # Remove horizontal
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ker, 1))
        detected_lines = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, horizontal_kernel, iterations=9
        )
        cnts = cv2.findContours(
            detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(image, [c], -1, (0,), (image.shape[0] // 81 + 2))

        return image

    def extract_cells(self, image: np.ndarray) -> list[np.ndarray]:
        """Extracts every cell in the grid.

        Args:
            image (np.ndarray): The processed image from which to extract the
                cells.

        Returns:
            list[np.ndarray]: A list of the extracted cells.
        """

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
            logger.debug(f"Show sample image on {t},{k}: {self.done}")

        logger.debug(f"Return list of grid cells: {type(final_grid)}")
        return final_grid

    def construct_board(
        self,
        model: Sequential,
        cells_list: list[np.ndarray],
        threshold: int = 400,
        er: tuple = (3, 3),
    ) -> list[int]:
        """Builds a sudoku board (in the form of a list of list of ints) from
            the given list of grid cells.

        Args:
            model (Sequential): The machine learning model that will predict
                the value of the digits present in the grid cells.
            cells_list (list[np.ndarray]): The list of grid cells.
            threshold (int, optional): The minimum number of white pixels to
                consider when searching for a number. Defaults to 400.
            er (int, optional): The number representing the size of the structuring element. Defaults to (3, 3).

        Returns:
            list[int]: The resulting sudoku board. Empty cells are represented
                with a zero value.
        """
        # list that will contain all the numbers
        board = []
        logger.debug(f"Create empty board – {type(board)}, {len(board)}")

        logger.debug(f"Iterating through grid of images...")
        for column in cells_list:
            temp_column = []  # column to append to board once filled
            for cell in column:
                cell = cell[3:-3, 3:-3]

                denoised_cell = ndimage.median_filter(cell, 3)

                # count white pixels, if below threshold, image is empty
                white_pixels = np.count_nonzero(denoised_cell)
                if white_pixels > threshold:

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, er)
                    cell = cv2.erode(cell, kernel)

                    # find contours of cell
                    contours, _ = cv2.findContours(
                        cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    # largest contour will be desired digit
                    digit_cntr = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(digit_cntr)

                    # crop image to digit
                    cell = cell[y : y + h, x : x + w]

                    # find padding to add to number to approach 32×32
                    t = abs(int(32 - cell.shape[0])) // 2
                    k = abs(int(32 - cell.shape[1])) // 2

                    # add black borders to image
                    cell = np.pad(cell, ((t, t), (k, k)), mode="constant")

                    # standardize image to input in network
                    mean_px = cell.mean().astype(np.float32)
                    std_px = cell.std().astype(np.float32)
                    cell = (cell - mean_px) / (std_px)

                    # resize to 32×32 for LeNet5
                    cell = cv2.resize(cell, (32, 32), cv2.INTER_CUBIC)

                    # make image a 4D tensor for network
                    cell = cell[np.newaxis, :, :, np.newaxis]

                    # predict number. verbose=0 cause output is useless
                    number = model.predict(cell, verbose=0).argmax()
                    temp_column.append(number)
                else:
                    temp_column.append(0)
            board.append(temp_column)

        logger.debug(f"Sudoku board extracted successfully: {len(board)}")
        return board

    def error_check(self, board: list[int]) -> list[int]:
        """Asks for user input and corrects the board accordingly

        Args:
            board (list[int]): The sudoku board.

        Returns:
            list[int]: The correct board.
        """
        correct = input("Has the board been correctly extracted? (Y/n) ")
        if correct in ["Y", "y", ""]:
            return board

        if correct in ["N", "n"]:
            row, col = [int(x) - 1 for x in input("Enter (row, col): ").split(", ")]
            print(board[row][col])

            replacement = int(input("Enter replacement: "))
            board[row][col] = replacement
            print(np.array(board))
            self.error_check(board)
        return board


# for debugging purposes
if __name__ == "__main__":
    import ImageProcessor
    from keras.models import load_model

    ipr = ImageProcessor.ImageProcessor(debug=True)
    sxt = SudokuExtractor(debug=True)

    # get image and process it
    original_image = ipr.read("./source/test_imgs/1.jpg")
    image = ipr.preprocess_image(original_image)

    # get grid contour and grid corners
    grid_contour = sxt.find_contour(image, original_image)  # var2 for debug
    corners = sxt.find_corners(grid_contour, original_image)  # var2 for debug

    # crop+warp image
    image = sxt.warp_image(image, corners)

    image = sxt.remove_grid(image)

    # rotate if necessary
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cells = sxt.extract_cells(image)

    print(np.count_nonzero(cells[7][5]))

    model = load_model("./source/LeNet5")
    # grid_original = sxt.construct_board(model, cells, 400)
    grid_original = sxt.construct_board(model, cells, 1200)
    # grid_original = sxt.construct_board(model, cells, 850, (1, 1))
    print(np.array(grid_original))


# remove logging so that no logging happens during app.py execution
# logger.remove()
