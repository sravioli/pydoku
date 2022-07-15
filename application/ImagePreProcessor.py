import cv2
from numpy import ndarray


class PreProcessor:
    def __init__(self):
        pass

    def process_image(self, image: ndarray) -> ndarray:
        """The image processor itself. What the function does: grayscale,
            Gaussian blur, adaptive threshold (binarization), inversion, morph,
            dilatation.

        Args:
            image (ndarray): The image to process.

        Returns:
            ndarray: The processed image.
        """
        # grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur image
        image = cv2.GaussianBlur(image, (9, 9), 0)

        # apply adaptive threshold: from grayscale to binary
        image = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # invert image (needed for cv2.findContours())
        image = cv2.bitwise_not(image, 0)

        # get rectangular kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # morph image to remove random noise
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # dilate image to increase border size
        result = cv2.dilate(image, kernel, iterations=1)

        return result
