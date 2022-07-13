import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


# constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH = "source/dataset"

# ratios+
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
IMG_DIMS = (32, 32)

# model
FILTERS = 60
FILTER1 = (5, 5)
FILTER2 = (3, 3)
POOL_SIZE = (2, 2)
NODES = 500

# training
BATCH_SIZE = 50
EPOCHS = 10
STEPS = 131
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# read dataset
def get_classes(path: str) -> list:
    # list directories inside PATH
    __dir_list = os.listdir(path)
    print(f"Total num of classes detected: {len(__dir_list)}")  # should be 10

    # store the number of folders. the latter is also the number of classes
    __classes = len(__dir_list)
    return __classes


# import dataset
def import_data(__classes: list, path: str) -> np.ndarray:
    # import all images and insert them inside a single list
    print("Importing classes...")

    images, class_ids = [], []
    # loop folders to obtain images inside them
    for _class in range(__classes):
        __img_list = os.listdir(f"{path}/{str(_class)}")

        # for every folder loop through the images
        for _img in __img_list:
            current_img = cv2.imread(f"{path}/{str(_class)}/{str(_img)}")

            # images are too big to be processed efficiently by the network thus
            # a resize is needed (180×180 -> 32×32)
            current_img = cv2.resize(current_img, IMG_DIMS)
            images.append(current_img)

            # save the corresponding ID (class ID) of each image
            class_ids.append(_class)
        print(_class, end=" ")
    print("\nDone")  # reset previous end command

    # convert lists to numpy arrays
    images = np.array(images)
    class_ids = np.array(class_ids)

    return images, class_ids


def get_samples(Y_train: np.ndarray, __classes: list) -> list:
    _samples = [len(np.where(Y_train == i)[0]) for i in range(__classes)]
    return _samples


def plot_samples(_samples: list, __classes: list) -> plt.figure:
    # visualize the data with histogram
    plt.figure(figsize=(10, 5))
    plt.bar(range(__classes), _samples)
    plt.title("Number of images for each class")
    plt.xlabel("Class ID")
    plt.ylabel("Number of images")
    plt.show()


def preprocess_imgs(image: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    image = cv2.equalizeHist(image)  # equalize img (distribute lighting evenly)
    image = image / 255  # normalize img (restrict from [0, 255] to [0, 1])

    return image


def preprocess(
    X_train: np.ndarray, X_test: np.ndarray, X_validation: np.ndarray
) -> np.ndarray:

    # preprocess all the images using map() &overwrite arrays
    X_train = np.array(list(map(preprocess_imgs, X_train)))
    X_test = np.array(list(map(preprocess_imgs, X_test)))
    X_validation = np.array(list(map(preprocess_imgs, X_validation)))

    # add depth to images
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    X_validation = X_validation.reshape(X_validation.shape + (1,))

    return X_train, X_test, X_validation


def get_model(
    filters_num: int,
    filter1: tuple,
    filter2: tuple,
    input_shape: tuple,
    pool_size: tuple,
    nodes_num: int,
    __classes: list,
) -> Sequential:

    # generate model
    model = Sequential()

    # add convolutional layers (×2)
    model.add(
        Conv2D(
            filters_num, filter1, input_shape=(input_shape + (1,)), activation="relu"
        )
    )
    model.add(Conv2D(filters_num, filter1, activation="relu"))

    # add pooling layer
    model.add(MaxPooling2D(pool_size=pool_size))

    # add two more convolutional layers with filter2 and less filters
    model.add(Conv2D(filters_num // 2, filter2, activation="relu"))
    model.add(Conv2D(filters_num // 2, filter2, activation="relu"))

    # add additional pooling layer
    model.add(MaxPooling2D(pool_size=pool_size))

    # add dropout layer at 50%
    model.add(Dropout(0.5))

    # add flatten layer
    model.add(Flatten())

    # add dense (×2) + dropout layers
    model.add(Dense(nodes_num, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(__classes, activation="softmax"))

    # compile model
    model.compile(
        Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def main() -> None:

    # get data
    __classes = get_classes(PATH)
    images, class_ids = import_data(__classes, PATH)

    # divide data
    # training = 0.8, testing = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(
        images, class_ids, test_size=TEST_RATIO
    )

    # training = 0.8, validation = 0.2
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=VALIDATION_RATIO
    )

    # preprocess data
    X_train, X_test, X_validation = preprocess(X_train, X_test, X_validation)

    # augment images
    DataGenerator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        rotation_range=0.1,
    )

    DataGenerator.fit(X_train)

    # one hot encode
    Y_train = to_categorical(Y_train, __classes)
    Y_test = to_categorical(Y_test, __classes)
    Y_validation = to_categorical(Y_validation, __classes)

    # get model + summary + to csv
    model = get_model(FILTERS, FILTER2, FILTER2, IMG_DIMS, POOL_SIZE, NODES, __classes)
    csv_logger = CSVLogger("source\model.csv")
    print(model.summary())

    # train the model
    history = model.fit(
        DataGenerator.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=STEPS,
        epochs=EPOCHS,
        validation_data=(X_validation, Y_validation),
        shuffle=1,
        callbacks=[csv_logger],
    )

    # save model
    model.save("../source/trained_model")


if __name__ == "__main__":
    main()
