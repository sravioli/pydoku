import numpy as np
import cv2

import keras
from keras.models import load_model

# from keras.utils.np_utils import probas_to_classes
import keras.utils

# settings
WIDTH, HEIGHT = 640, 480


model = load_model("trained_model.h5", compile=True)

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)


def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    img = cv2.equalizeHist(img)  # equalize img (distribute lighting evenly)
    img = img / 255  # normalize img (restrict value from [0, 255] to [0, 1])

    return img


if not cap.isOpened():
    raise IOError("Cannot open webcam")


while True:
    success, original_img = cap.read()
    cv2.imshow("input", original_img)

    img = preprocess(original_img)
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = img.reshape(1, 32, 32, 1)

    # predict
    class_index = model.predict(img)
    # proba = np_utils.probas_to_classes(class_index)
    proba = np.argmax(class_index, axis=1)
    print(proba[0])

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
