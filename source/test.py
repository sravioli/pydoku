import numpy as np
import cv2

from keras.models import load_model
from sklearn.preprocessing import scale

from training import preprocess_img

# constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WIDTH, HEIGHT = 640, 480
THRESHOLD = 75.00
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def process_img(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    image = cv2.resize(image, (32, 32))
    image = image.reshape(1, 32, 32, 1)
    return image


def predict(image: np.ndarray) -> tuple:
    __class_index = model.predict(image)
    _index = np.argmax(__class_index, axis=1)
    _probability = round(np.amax(__class_index) * 100, 2)

    return (_index[0], _probability)


def display(original_img: np.ndarray, _index: int, _probability: float) -> None:
    cv2.putText(
        original_img,  # image
        str(_index) + " " + str(_probability),  # text
        (50, 50),  # point
        cv2.FONT_HERSHEY_COMPLEX,  # font face
        1.5,  # font scale
        (0, 0, 255),  # color
        1,  # thickness
        cv2.LINE_AA,  # line type
    )


def mainloop(cap) -> None:
    # open webcam
    while True:
        ret, original_img = cap.read()

        if not ret:
            print("Cannot receive frame. Exiting...")
            break

        image = preprocess_img(original_img)
        image = process_img(image)

        # predict
        index, probability = predict(image)
        print(index, probability)

        if probability > THRESHOLD:
            display(original_img, index, probability)

        cv2.imshow("Webcam Input", original_img)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # load model
    model = load_model("./source/trained_model", compile=True)

    # initiate capture object
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    # check
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    mainloop(cap)


if __name__ == "__main__":
    main()
