from pathlib import Path, WindowsPath
import os


import numpy as np
import cv2 as cv

from halo import Halo

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


class DataReader:
    def __init__(self, folder_name: str):
        """A utility class to read data from a folder and import it into the
            script.

        Args:
            folder_name (str): The name of the folder containing the data. It
                is supposed that the folder is inside the directory of the
                script.
        """
        self.folder_name = folder_name

    def get_path(self) -> WindowsPath:
        """Retrieves the absolute path relative to the self.folder_name.

        Returns:
            WindowsPath: The absolute path of the dataset.
        """
        return Path(__file__).parent / self.folder_name

    def get_classes(self) -> int:
        """Retrieves the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        __path = self.get_path()

        __dir_list = os.listdir(__path)
        _classes = len(__dir_list)

        return _classes

    def import_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Imports the data from the given dataset into the script.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple of arrays containing
                respectively the images list and the class IDs.
        """
        _spinner = Halo(  # define spinner
            text=f"Reading data from ./{self.folder_name}/",
            spinner={
                "interval": 80,
                "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
            },
            color="magenta",
        )

        __path = self.get_path()
        classes = self.get_classes()
        print(f"Detected \033[0;34m{classes}\033[0m classes")  # should be 10

        _spinner.start()  # start spinner

        images, class_ids = [], []
        # loop through folders and retrieve images
        for folder in range(classes):
            __img_list = os.listdir(__path / str(folder))

            # loop through images and create list
            for image in __img_list:
                current_img = cv.imread(f"{__path}/{str(folder)}/{str(image)}")
                # resize image to 32×32
                current_img = cv.resize(current_img, (32, 32))
                images.append(current_img)

                # save the class IDs (equal to folder number)s
                class_ids.append(folder)

        # convert to array
        images, class_ids = np.array(images), np.array(class_ids)

        _spinner.stop()  # stop spinner
        print("\033[0;32m✓\033[0m Done")

        return images, class_ids


class PreProcessor:
    def __init__(
        self, training: np.ndarray, testing: np.ndarray, validating: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A utility class for preprocessing images.

        Args:
            training (np.ndarray): The (X/Y)_training data.
            testing (np.ndarray): The (X/Y)_testing data.
            validating (np.ndarray): The (X/Y)_validating data.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
                preprocessed data in the same order as the input. For
                example: (<num>, 32, 32, 1).
        """
        self.training = training
        self.testing = testing
        self.validating = validating

    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Processes a single image (grayscale, equalize, normalize).

        Args:
            image (np.ndarray): The single image to process.

        Returns:
            np.ndarray: The single image processed
        """
        # convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # equalize img (distribute lighting evenly)
        image = cv.equalizeHist(image)
        # normalize image (restrict color from [0, 255] to [0, 1])
        image = cv.normalize(image, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)

        return image

    def process_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The actual data processor. Applies self.process_image() to all images in input, adds dimension to the array

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the
                preprocessed data in the same order as the input. For
                example: (<num>, 32, 32, 1).
        """
        # cannot figure out how to loop (for would not work cv throws error)

        # preprocess the images
        self.training = np.array(list(map(self.process_image, self.training)))
        self.testing = np.array(list(map(self.process_image, self.testing)))
        self.validating = np.array(list(map(self.process_image, self.validating)))

        # add depth to images
        self.training = self.training.reshape(self.training.shape + (1,))
        self.testing = self.testing.reshape(self.testing.shape + (1,))
        self.validating = self.validating.reshape(self.validating.shape + (1,))

        return self.training, self.testing, self.validating


class PlotDistribution:
    def __init__(self, Y_train: np.ndarray, classes: int) -> plt.figure:
        """A utility class for generating a histogram of the distribution of
            the splitted data.

        Args:
            Y_train (np.ndarray): The Y_training data.
            classes (int): The number of classes.

        Returns:
            plt.figure.Figure: A matplotlib.figure.Figure object containing the
                histogram.
        """
        self.Y_train = Y_train
        self.classes = classes

    def get_samples(self) -> list:
        """Retrieves a list of samples from data.

        Returns:
            list: The list of samples present in each class.
        """
        return [len(np.where(self.Y_train == i)[0]) for i in range(self.classes)]

    def plot_samples(self) -> plt.figure:
        """Plots the histogram of the distribution of the samples in each class.

        Returns:
            plt.figure.Figure: A matplotlib.figure.Figure object containing the
                histogram.
        """
        samples = self.get_samples()

        # visualize the data with histogram
        plt.figure(figsize=(10, 5))
        plt.bar(range(self.classes), samples)
        plt.title("Number of images for each class")
        plt.xlabel("Class ID")
        plt.ylabel("Number of images")
        plt.show()


def augment_images(train: np.ndarray) -> ImageDataGenerator:
    DataGenerator = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        shear_range=0.1,
        rotation_range=0.1,
    )
    DataGenerator.fit(train)

    return DataGenerator


def get_model(
    filters_num: int,
    filter1: tuple,
    filter2: tuple,
    # input_shape: tuple,
    pool_size: tuple,
    nodes_num: int,
    __classes: list,
) -> Sequential:

    # generate model
    model = Sequential()

    # add convolutional layers (×2)
    model.add(
        Conv2D(filters_num, filter1, input_shape=((32, 32) + (1,)), activation="relu")
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


def parent_path() -> WindowsPath:
    return Path(__file__).parent


def model_logger(filename: str) -> CSVLogger:
    path = parent_path()
    logger = CSVLogger(f"{path}/{filename}")

    return logger
