#!/usr/bin/env python3

from loguru import logger
from source import ImageProcessor, SudokuExtractor, SudokuSolver
from keras.models import load_model
import numpy as np

import cv2

logger.remove()


def main():
    # load classes
    ipr = ImageProcessor.ImageProcessor()
    sxt = SudokuExtractor.SudokuExtractor()
    slv = SudokuSolver.SudokuSolver()

    # load preferred model
    model = load_model("./source/LeNet5")

    # import image
    original_image = ipr.read("./source/test_imgs/1.jpg")

    # preprocess image
    image = ipr.preprocess_image(original_image)

    # find sudoku grid contours
    sudoku_contour = sxt.find_contour(image)

    # find sudoku grid corners
    grid_corners = sxt.find_corners(sudoku_contour)

    # warp image (this also crops it)
    warped_image = sxt.warp_image(image, grid_corners)

    # remove grid from image
    image = sxt.remove_grid(warped_image)

    # depending on the image, a rotation may be necessary
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # extract every sudoku cell from the warped image
    cells = sxt.extract_cells(image)

    # construct grid of numbers from the list of cells
    board = sxt.construct_board(model, cells, 1200)

    # can print unsolved grid
    slv.print(board)

    sxt.error_check(board)

    slv.solve(board)
    slv.print(board)


if __name__ == "__main__":
    main()
