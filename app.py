#!/usr/bin/env python3

from source import ImageProcessor, SudokuExtractor, SudokuSolver
from keras.models import load_model

model = load_model("./source/LeNet5")

ipr = ImageProcessor.ImageProcessor()
sxt = SudokuExtractor.SudokuExtractor()
slv = SudokuSolver.SudokuSolver()

# get image and process it
original_image = ipr.read("./source/test_imgs/sudoku.jpg")
image = ipr.preprocess_image(original_image)

# find grid contour and corners
grid_contour = sxt.find_contour(image)
corners = sxt.find_corners(grid_contour)

# crop+warp image
image = sxt.warp_image(image, corners)

# rotate if needed
# image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

cells = sxt.extract_cells(image)
grid = sxt.construct_grid(model, cells)

slv.solve(grid)
slv.print(grid)
