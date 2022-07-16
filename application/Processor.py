from typing import Union

import numpy as np
import operator

import cv2

from tensorflow import convert_to_tensor


class ProcessorHelper:
    def __init__(self):
        pass

    def find_vertex(
        # self,
        polygon: np.ndarray,
        limit,
        operation,
    ) -> tuple:
        limits, operations = ["min", "max"], ["add", "subtract"]
        if limit not in limits:
            raise ValueError(f"Invalid function type expected one of {limits}.")
        if operation not in operations:
            raise ValueError(f"Invalid function type expected one of {operations}.")

        if limit == limits[0]:
            limit = min
        if operation == operations[0]:
            operation = np.add

        limit, operation = max, np.subtract

        # for ex. in finding the bottom left corner of a polygon, this will
        # have the smallest (x - y) value.
        section, _ = limit(
            enumerate([operation(pt[0][0], pt[0][1]) for pt in polygon]),
            key=operator.itemgetter(1),
        )

        # this should be the coords of the vertex
        return polygon[section][0][0], polygon[section][0][1]

    def draw_vertex(image: np.ndarray, points: tuple) -> None:
        cv2.circle(image, points, 7, (0, 255, 0), cv2.FILLED)

    def get_grid_lines(
        self, image: np.ndarray, shape_location: int = 0 | 1, length: int = 10
    ) -> np.ndarray:
        if shape_location not in [0, 1]:
            raise ValueError("The location of the shape must be either 0 or 1.")

        clone = image.copy()

        # if the lines are the horizontal ones, shape_location is 1, else 0
        row_or_col = clone.shape[shape_location]

        # get distance between lines
        size = row_or_col // length

        # get appropriate kernel
        if shape_location == 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

        # erode and dilate lines
        clone = cv2.erode(clone, kernel)
        clone = cv2.dilate(clone, kernel)

        return clone

    # lines = points
    def draw_lines(self, image: np.ndarray, lines: np.ndarray) -> np.ndarray:
        clone = image.copy()
        lines = np.squeeze(lines)

        for rho, theta in lines:
            # find where line stretches and draw it
            x1, y1 = (
                int((rho * np.cos(theta)) + 1000 * (-np.sin(theta))),
                int((rho * np.sin(theta)) + 1000 * np.cos(theta)),
            )
            x2, y2 = (
                int((rho * np.cos(theta)) - 1000 * (-np.sin(theta))),
                int((rho * np.sin(theta)) - 1000 * np.cos(theta)),
            )
            # backup
            # a, b = np.cos(theta), np.sin(theta)
            # x0, y0 = a * rho, b * rho
            # x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
            # x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
            cv2.line(clone, (x1, y1), (x2, y2), (255, 255, 255), 4)
        return clone

    # img = square
    def clean_square(self, square: np.ndarray) -> tuple[np.ndarray, bool]:
        # basically check if is black
        if np.isclose(square, 0).sum() / (square.shape[0] * square.shape[1]) >= 0.95:
            return np.zeros_like(square), False

        # if some white is present in the center of the square, the latter is
        # an edge. This is unwanted
        height, width = square.shape
        mid = width // 2
        if (
            np.isclose(
                square[:, int(mid - width * 0.4) : int(mid + width * 0.4)], 0
            ).sum()
            / (2 * width * 0.4 * height)
            >= 0.90
        ):
            return np.zeros_like(square), False

        # center image ad extract numbers
        contours, _ = cv2.findContours(
            square,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x, y, w, h = cv2.boundingRect(contours[0])

        start_x = (width - w) // 2
        start_y = (height - h) // 2
        new_square = np.zeros_like(square)
        new_square[start_y : start_y + h, start_x : start_x + w] = square[
            y : y + h, x : x + w
        ]
        return new_square, True


class Processor:
    def __init__(self):
        pass

    # img = processed_image, original = image_corners
    def find_contours(
        self, processed_image: np.ndarray, image_corners: np.ndarray
    ) -> list[tuple, tuple, tuple, tuple]:
        # contours are the the boundaries of a shape with same intensity.
        # find contours in image w/ threshold applied (_ is hierarchy).
        # cv2.findContours() finds the coordinates of the boundary of an image
        contours, _ = cv2.findContours(
            processed_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,  # save only the two end points
        )

        # sort by largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        polygon = None
        # check that the polygon is the correct one
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, closed=True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, closed=True)
            corners_number = len(approx)

            if corners_number == 4 and area > 1000:
                polygon = contour
                break

        if polygon is not None:
            # find the vertices of the polygon
            top_left = ProcessorHelper.find_vertex(polygon, "min", "add")
            top_right = ProcessorHelper.find_vertex(
                polygon,
                "max",
                "subtract",
            )
            bot_left = ProcessorHelper.find_vertex(
                polygon,
                "min",
                "subtract",
            )
            bot_right = ProcessorHelper.find_vertex(polygon, "max", "add")

            # check if it's a square, if not, trash it
            if bot_right[1] - top_right[1] == 0:
                return []
            if not (
                0.95
                < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1]))
                < 1.05
            ):
                return []

            cv2.drawContours(image_corners, [polygon], 0, (0, 0, 255), 3)

            # draw corresponding circles
            [
                ProcessorHelper.draw_vertex(x, image_corners)
                for x in [top_left, top_right, bot_right, bot_left]
            ]

            return [top_left, top_right, bot_right, bot_left]
        return []

    # corners = image_corners (np.ndarray), original = image (np.ndarray)
    def warp_image(
        self,
        image_corners: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        # warp points
        image_corners = np.array(image_corners, dtype="float32")
        top_left, top_right, bot_right, bot_left = image_corners

        # compute best side width, warp into square so height = length
        width = int(
            max(
                [
                    np.linalg.norm(top_right - bot_right),
                    np.linalg.norm(top_left - bot_left),
                    np.linalg.norm(bot_right - bot_left),
                    np.linalg.norm(top_left - top_right),
                ]
            )
        )

        # create array with shows, top_left, top_right, ...
        mapping = np.array(
            [[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]],
            dtype="float32",
        )

        warp_matrix = cv2.getPerspectiveTransform(image_corners, mapping)

        return cv2.warpPerspective(image, warp_matrix, (width, width))  # , warp_matrix

    def get_grid_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        horizontal_lines = ProcessorHelper.get_grid_lines(image, 1)
        vertical_lines = ProcessorHelper.get_grid_lines(image, 0)
        return horizontal_lines, vertical_lines

    def create_grid_mask(
        horizontal_lines: np.ndarray, vertical_lines: np.ndarray
    ) -> np.ndarray:
        # combine horizontal and vertical lines to make a grid
        grid = cv2.add(horizontal_lines, vertical_lines)

        # threshold and dilate the grid to cover more area
        grid = cv2.adaptiveThreshold(
            grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2
        )
        grid = cv2.dilate(
            grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2
        )

        # find lines in binary images with Hough transform. Uses radians
        points = cv2.HoughLines(grid, 0.3, np.pi / 90, 200)

        lines = ProcessorHelper.draw_lines(grid, points)

        # extract lines
        mask = cv2.bitwise_not(lines)
        return mask

    def split_in_squares(warped_image: np.ndarray) -> list[np.ndarray]:
        squares = []
        width = warped_image.shape[0] // 9

        # find each square assuming they are of same size
        for j in range(9):
            for i in range(9):
                # top left corner of bounding box
                p1 = i * width, j * width

                # bottom right corner of bounding box
                p2 = (
                    (i + 1) * width,
                    (j + 1) * width,
                )
                squares.append(warped_image[p1[1] : p2[1], p1[0] : p2[0]])
        return squares

    def clean_squares(squares: list[np.ndarray]) -> list[np.ndarray]:
        clean_squares = []
        i = 0

        for square in squares:
            new_image, is_number = ProcessorHelper.clean_square(square)

            if is_number:
                clean_squares.append(new_image)
                i += 1  # do we need this?
            else:
                clean_squares.append(0)
        return clean_squares

    def recognize_digits(grid_squares, model):
        s = ""
        formatted_squares = []
        location_of_zeroes = set()

        blank = np.zeros_like(cv2.resize(grid_squares[0], (32, 32)))

        for i in range(len(grid_squares)):
            if type(grid_squares[i]) == int:  # isn't isinstance better?
                location_of_zeroes.add(i)
                formatted_squares.append(blank)
            else:
                image = cv2.resize(grid_squares[i], (32, 32))
                formatted_squares.append(image)

        formatted_squares = np.array(formatted_squares)
        predictions = list(map(np.argmax, model(convert_to_tensor(formatted_squares))))

        for i in range(len(predictions)):
            if i in location_of_zeroes:
                s += "0"
            else:
                s += str(predictions[i] + 1)
        return s

    # warped_img = warped_image, squares_processed = grid_squares
    # index = k
    def draw_digits_on_warped(
        warped_img: np.ndarray, solved_puzzle: str, squares_processed: np.ndarray
    ) -> np.ndarray:
        width = warped_img.shape[0] // 9
        image_w_text = np.zeros_like(warped_img)

        index = 0
        # find each square, assuming they have same size
        for j in range(9):
            for i in range(9):
                if type(squares_processed[index]) == int:
                    # top left corner of bounding box
                    p1 = i * width, j * width

                    # bottom right corner of bounding box
                    p2 = (
                        (i + 1) * width,
                        (j + 1) * width,
                    )

                    center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    text_size, _ = cv2.getTextSize(
                        str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4
                    )
                    text_origin = (
                        center[0] - text_size[0] // 2,
                        center[1] + text_size[1] // 2,
                    )

                    cv2.putText(
                        warped_img,
                        str(solved_puzzle[index]),
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                    )
            index += 1
        return image_w_text

    def unwarp_image(
        warped_image: np.ndarray,
        image_result: np.ndarray,
        corners: list[tuple],
        time,
    ) -> np.ndarray:
        corners = np.array(corners)

        height, width = warped_image.shape[0], warped_image.shape[1]
        pts_source = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
            dtype="float32",
        )
        h, status = cv2.findHomography(pts_source, corners)
        warped = cv2.warpPerspective(
            warped_image, h, (image_result.shape[1], image_result.shape[0])
        )
        cv2.fillConvexPoly(image_result, corners, 0, 16)

        dst_img = cv2.add(image_result, warped)

        dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
        cv2.putText(
            dst_img,
            time,
            (dst_img_width - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2,
        )

        return dst_img
