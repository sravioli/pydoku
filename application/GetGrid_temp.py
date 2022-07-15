import operator

import cv2 as cv
import numpy as np


# helps with the process of finding the grid
class GridProcessorHelper:
    def find_vertexes(
        polygon,
        limit: function = min | max,
        operation: function = np.add | np.subtract,
    ) -> tuple:

        # for ex. in finding the bottom left corner of a polygon, this will
        # have the smallest (x - y) value.
        section, _ = limit(
            enumerate([operation(pt[0][0], pt[0][1]) for pt in polygon]),
            key=operator.itemgetter(1),
        )

        # return coords?? ig idk
        return polygon[section][0][0], polygon[section][0][1]

    def draw_vertexes(original, pts: tuple):
        # wtf is original
        cv.circle(original, pts, 7, (0, 255, 0), cv.FILLED)

    def check_cell(cell: list) -> tuple[np.ndarray, bool]:

        # check if grid cell contains no number
        if np.isclose(cell, 0).sum() / (cell.shape[0] * cell.shape[1]) >= 0.95:
            return np.zeros_like(cell), False

        # it little white is present in the region around the center, this
        # signifies that the cell is an edge
        height, width = cell.shape
        mid = width // 2
        if (
            np.isclose(
                cell[:, int(mid - width * 0.4) : int(mid + width * 0.4)], 0
            ).sum()
            / (2 * width * 0.4 * height)
            >= 0.90
        ):
            return np.zeros_like(cell), False

        # center image
        contours, _ = cv.findContours(cell, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv.contourArea, reverse=True)
        x, y, w, h = cv.boundingRect(contours[0])

        start_x = (width - w) // 2
        start_y = (height - h) // 2
        new_img = np.zeros_like(cell)
        new_img[start_y : start_y + h, start_x : start_x + w] = cell[
            y : y + h, x : x + w
        ]

        return new_img, True
