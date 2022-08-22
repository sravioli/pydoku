# logging
from loguru import logger
import sys

# ImagePreprocessor
from pathlib import Path
import numpy as np
import cv2


logger.remove()
logger_format = (
    "<green>{time:DD/MM/YYYY – HH:mm:ss}</green> "
    + "| <lvl>{level: <8}</lvl> "
    + "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    + "– <lvl>{message}</lvl>"
)
logger.add(sys.stdout, colorize=True, format=logger_format)


class ImageProcessor:
    def __init__(self, debug: bool = False) -> np.ndarray:
        """Utility class to read and preprocess an input image.

        Args:
            debug (bool, optional): Whether or not to display the various
                steps. Defaults to False.
        """
        self.debug = debug
        self.done = "\033[92m✓\033[39m"

    def get_path(self, image_path: str) -> str:
        """Retrieves the absolute path of "pydoku" folder. This means that it's
            supposed that all the files are present in the aforementioned
            folder.

        Args:
            image_path (str): The relative (to "pydoku") path of the input
                image.

        Returns:
            str: The absolute path of the image.
        """
        path = Path(__file__).absolute()
        logger.debug(f"Input (absolute) – {type(path)}")

        # convert to str, split at "\\", get index of "pydoku", assign list
        t_index = (path_list := str(path).split("\\")).index("pydoku")
        logger.debug(f"Find target index: {t_index} – '{path_list[t_index]}'")

        # remove elements after "pydoku", join list to str
        final_path = "\\".join(path_list[: t_index + 1])
        logger.debug(f"Stripped path (absolute) – {type(final_path)}")

        final_path = str(Path(final_path + image_path))

        logger.debug(f"Return final path – {type(final_path)}")
        return final_path

    # read image
    def read(self, image_path: str) -> np.ndarray:
        """Reads the image and returns it.

        Args:
            image_path (str): The relative (to "pydoku") path of the input
                image.

        Returns:
            np.ndarray: The imported image.
        """
        absolute_image_path = self.get_path(image_path)
        logger.debug(f"Input (absolute path) – {type(absolute_image_path)}")

        image = cv2.imread(absolute_image_path)
        logger.debug(f"Read image: {self.done}")

        if self.debug:
            cv2.imshow("DEBUG: INPUT IMAGE", cv2.resize(image, (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        logger.debug(f"Return read image – {type(image)}")
        return image

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Processes the image. Applies: grayscale, Gaussian blur, adaptive
            threshold, then: inverts image, morphs image, dilates image.

        Args:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The processed image.
        """
        # grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        logger.info(f"Apply grayscale: {self.done}")

        # apply Gaussian blur. Kernel must be positive and square
        image = cv2.GaussianBlur(image.copy(), (9, 9), 0)
        logger.info(f"Apply Gaussian blur: {self.done}")

        # threshold image to segment it
        image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
        logger.info(f"Apply threshold: {self.done}")

        # invert image
        image = cv2.bitwise_not(image, image)
        logger.info(f"Invert image: {self.done}")

        # morph image to reduce random noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        logger.info(f"Morph image: {self.done}")

        # dilate image since Gaussian blur and thresholding shrink it
        kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
            np.uint8,
        )
        image = cv2.dilate(image, kernel)
        logger.info(f"Dilate image: {self.done}")

        if self.debug:
            cv2.imshow("DEBUG: PROCESSED IMAGE", cv2.resize(image, (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        logger.debug(f"Return processed image – {type(image)}")
        return image


# for debugging purposes
if __name__ == "__main__":
    ipr = ImageProcessor(debug=True)

    # read image and process it
    image = ipr.read("/source/test_imgs/sudoku-pencil.jpg")
    process = ipr.preprocess_image(image)
