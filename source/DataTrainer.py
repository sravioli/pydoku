# probably shebang needed

# for logging
import enum
from cv2 import log
from loguru import logger
import sys

# DataTrainer
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

from keras.models import Sequential


from keras.callbacks import ReduceLROnPlateau
from keras.layers import Conv2D, BatchNormalization, Activation
from keras.layers import MaxPooling2D, Dropout, Flatten, Dense
from keras.regularizers import L2
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


import cv2
import numpy as np


logger.remove()
logger_format = (
    "<green>{time:DD/MM/YYYY – HH:mm:ss}</green> "
    + "| <lvl>{level: <8}</lvl> "
    + "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    + "– <lvl>{message}</lvl>"
)
logger.add(
    sys.stdout,
    colorize=True,
    format=logger_format,
)

# https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392


class DataTrainer:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.done = "\033[92m✓\033[39m"
        pass

    def load_mnist(self):
        # define MNIST dataset
        mnist = keras.datasets.mnist
        logger.debug(f"Define MNIST - {type(mnist)}")

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        logger.debug(f"Import dataset: {x_train.shape} – {type(x_test)}")

        if self.debug:
            sample = np.random.randint(60_000)
            logger.debug(
                f"Show random sample #{sample} before normalization: {self.done}"
            )
            cv2.imshow(
                f"DEBUG: SAMPLE #{sample} BEFORE NORMALIZATION",
                cv2.resize(x_train[sample], (320, 320)),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # scale images to [0, 1] range
        x_train = x_train.astype(np.float32) / 255
        x_test = x_test.astype(np.float32) / 255

        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        logger.debug(f"Check image dimensions: {x_train.shape}, {x_test.shape}")

        # normalize images
        mean_train, mean_test = (
            x_train.mean().astype(np.float32),
            x_test.mean().astype(np.float32),
        )
        std_train, std_test = (
            x_train.std().astype(np.float32),
            x_test.std().astype(np.float32),
        )
        logger.debug(
            f"Find mean and std: {mean_train, mean_test} – {std_train, std_test}"
        )

        x_train, x_test = (
            ((x_train - mean_train) / std_train),
            ((x_test - mean_test) / std_test),
        )
        logger.debug(f"Normalize images: {self.done}")

        if self.debug:
            logger.debug(
                f"Show random sample #{sample} after normalization: {self.done}"
            )
            cv2.imshow(
                f"DEBUG: SAMPLE #{sample} AFTER NORMALIZATION",
                cv2.resize(x_train[sample], (320, 320)),
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return (x_train, y_train), (x_test, y_test)

    def get_LeNet5(
        self,
        train: tuple[np.ndarray, np.ndarray],
        test: tuple[np.ndarray, np.ndarray],
    ) -> Sequential:
        # divide input data
        x_train, y_train = train
        x_test, y_test = test

        # get validation data
        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train, y_train, test_size=0.1, random_state=2
        )

        # create enhanced LeNet5 model
        model = Sequential(
            [
                # 1 – Convolutional2D, 32@32×32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Conv2D(
                    32,
                    (5, 5),
                    strides=1,
                    activation="relu",
                    input_shape=(32, 32, 1),
                    kernel_regularizer=L2(0.0005),
                ),
                # 2 – Convolutional2D, 32@28×28 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Conv2D(32, (5, 5), strides=1, use_bias=False),
                BatchNormalization(),  # maintain mean to 0, std to 1
                Activation("relu"),  # apply activation function to output.
                # 3. Layer – MaxPooling, 32@14x14 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                MaxPooling2D(pool_size=2, strides=2),
                Dropout(0.25),  # sets units to 0 with frequency of 0.25, random
                # 4 – Convolutional2D, 64@12×12 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Conv2D(
                    64,
                    (3, 3),
                    strides=1,
                    activation="relu",
                    kernel_regularizer=L2(0.0005),
                ),
                # 5 – Convolutional2D, 64@10×10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False),
                BatchNormalization(),  # maintain mean to 0, std to 1
                Activation("relu"),  # apply activation function to output.
                # 6 – MaxPooling2D, 64@5×5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                MaxPooling2D(pool_size=2, strides=2),
                Dropout(0.25),  # randomly set units to 0 with frequency of 0.25
                Flatten(),  # Flatten input. Does not affect the batch size.
                # 7 – Dense, 1×256 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Dense(units=256, use_bias=False),
                BatchNormalization(),  # maintain mean to 0, std to 1
                Activation("relu"),  # apply activation function to output.
                # 8 – Dense, 1×128 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Dense(units=128, use_bias=False),
                BatchNormalization(),  # maintain mean to 0, std to 1
                Activation("relu"),  # apply activation function to output.
                # 9 – Dense, 1×84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Dense(units=84, use_bias=False),
                BatchNormalization(),  # maintain mean to 0, std to 1
                Activation("relu"),  # apply activation function to output.
                Dropout(0.25),
                # 10 – Dense 1×10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                Dense(units=10, activation="softmax"),
            ]
        )

        model.compile(
            Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # alter train images
        data_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by dataset std
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # rotate images in the range (deg, 0-180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # shift images horizontally (frac of tot w)
            height_shift_range=0.1,  # shift images vertically (frac of tot h)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
        )
        data_gen.fit(x_train)

        # by using a Variable Learning Rate a significant performance increase
        # is registered. As soon as the model detects 'stagnation', the LR is
        # decreased by a factor of 0.2
        variable_learning_rate = ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=0.00001,
        )

        epochs, batch_size = 30, 86
        history = model.fit(
            data_gen.flow(x_train, x_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_validation, y_validation),
            verbose=2,
            steps_per_epoch=x_train.shape[0] // batch_size,
            callbacks=[variable_learning_rate],
        )

        return model


if __name__ == "__main__":
    dt = DataTrainer(debug=True)
    (x_train, y_train), (x_test, y_test) = dt.load_mnist()
    LeNet5 = dt.get_LeNet5((x_train, y_train), (x_test, y_test))
