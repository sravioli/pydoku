import time as t

import cv2

from pathlib import Path
from keras.models import load_model

# from ... import
from application import ImagePreProcessor

from application import Processor

# constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MODEL = "./source/trained_model"
ABSPATH = Path(__file__).parent / MODEL

WIDTH, HEIGHT = 960, 720
BRIGHTNESS = 150
FRAME_RATE = 30
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# setup video capture
cap = cv2.VideoCapture(0)  # num is camera on device

# set windows resolution & brightness
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(10, BRIGHTNESS)

# load model
model = load_model(ABSPATH, compile=True)

# setup classes from imports
pre = ImagePreProcessor.PreProcessor()  # preprocessor
prc = Processor.Processor()  # extractor

previous_time = 0
seen = dict()

# raise error if webcam is not available
if not cap.isOpened():
    raise IOError("Unable to open webcam.")

# initiate video capture
while True:
    elapsed_time = t.time() - previous_time

    # capture frame-by-frame
    success, image = cap.read()

    if elapsed_time > 1.0 / FRAME_RATE:
        previous_time = t.time()

        # copy image print result on one, process the other
        image_result, image_corners = image.copy(), image.copy()

        # process image & find corners (idk bout that one)
        processed_image = pre.process_image(image)
        corners = prc.find_contours(processed_image, image_corners)

        # check if list of corners is empty
        if corners:
            # warp and then process image
            warped_image = prc.warp_image(corners, image)  # _ is matrix
            warped_image = pre.process_image(warped_image)

            # retrieve grid lines
            vertical_lines, horizontal_lines = prc.get_grid_lines(warped_image)

            # mask image and retrieve numbers
            mask = prc.create_grid_mask(vertical_lines, horizontal_lines)
            numbers = cv2.bitwise_and(warped_image, mask)

            # split grid into squares
            grid_squares = prc.split_in_squares(numbers)
            grid_squares = prc.clean_squares(grid_squares)

            guess_squares = prc.recognize_digits(grid_squares, model)

    # display the solved sudoku
    cv2.imshow("Webcam", image_result)

    # wait for Esc or Q key to terminate process
    key = cv2.waitKey(1)
    if key == 27 or key == 113:
        break

cv2.destroyAllWindows()
cap.release()
