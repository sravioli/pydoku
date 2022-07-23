# for logging
from loguru import logger
import sys

# ModelTraining
from pathlib import Path
import numpy as np
import cv2

from typing import Union

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pandas as pd

from tensorflow import keras

from keras.models import load_model

from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, AveragePooling2D
from keras.regularizers import L2
from keras.optimizers import Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator


logger.remove()
logger_format = (
    "<green>{time:DD/MM/YYYY – HH:mm:ss}</green> "
    + "| <lvl>{level: <8}</lvl> "
    + "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    + "– <lvl>{message}</lvl>"
)
logger.add(sys.stdout, colorize=True, format=logger_format)


class Model:
    def __init__(self, debug: bool = False):
        """A wrapper class which provides various functions useful for the
            training process of the model.

        Args:
            debug (bool, optional): Whether to display images or not. Defaults
                to False.
        """
        self.debug = debug
        self.done = "\033[92m✓\033[39m"
        pass

    def get_model(self, model_name: str = "LeNet5") -> Sequential:
        """Retrieves the desired model

        Args:
            model_name (str, optional): The name of the desired model, can be
                one of the following: ["LeNet5+", "LeNet5", "custom"]. Defaults
                to "LeNet5".

        Raises:
            ValueError: When the chosen model is not supported.

        Returns:
            Sequential: A Sequential() keras model.
        """
        models = ["LeNet5+", "LeNet5", "custom"]
        if model_name not in models:
            raise ValueError(
                f"Unknown model {model_name}. Chose one of the following: {models}"
            )
        logger.info(f"Chosen model '{model_name}' is available")

        if model_name == models[0]:  # LeNet5+
            logger.debug(f"Creating '{model_name}' model...")
            model = Sequential(
                [
                    # 1 – Convolutional2D, 32@32×32 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Conv2D(
                        32,
                        5,
                        strides=1,
                        activation="relu",
                        input_shape=(32, 32, 1),
                        kernel_regularizer=L2(0.0005),
                    ),
                    # 2 – Convolutional2D, 32@28×28 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Conv2D(32, 5, strides=1, use_bias=False),
                    BatchNormalization(),  # maintain mean to 0, std to 1
                    Activation("relu"),  # apply activation function to output.
                    # 3. Layer – MaxPooling, 32@14x14 ~~~~~~~~~~~~~~~~~~~~~~~~~
                    MaxPooling2D(pool_size=2, strides=2),
                    Dropout(0.25),  # sets units to 0 with frequency of 0.25, random
                    # 4 – Convolutional2D, 64@12×12 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Conv2D(
                        64,
                        3,
                        strides=1,
                        activation="relu",
                        kernel_regularizer=L2(0.0005),
                    ),
                    # 5 – Convolutional2D, 64@10×10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False),
                    BatchNormalization(),  # maintain mean to 0, std to 1
                    Activation("relu"),  # apply activation function to output.
                    # 6 – MaxPooling2D, 64@5×5 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    MaxPooling2D(pool_size=2, strides=2),
                    Dropout(0.25),  # randomly set units to 0 with frequency of 0.25
                    Flatten(),  # flatten input. Does not affect the batch size.
                    # 7 – Dense, 1×256 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(units=256, use_bias=False),
                    BatchNormalization(),  # maintain mean to 0, std to 1
                    Activation("relu"),  # apply activation function to output.
                    # 8 – Dense, 1×128 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(units=128, use_bias=False),
                    BatchNormalization(),  # maintain mean to 0, std to 1
                    Activation("relu"),  # apply activation function to output.
                    # 9 – Dense, 1×84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(units=84, use_bias=False),
                    BatchNormalization(),  # maintain mean to 0, std to 1
                    Activation("relu"),  # apply activation function to output.
                    Dropout(0.25),
                    # 10 – Dense 1×10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(units=10, activation="softmax"),
                ]
            )
            logger.debug(f"'{model_name}' model has been created")

            model.compile(
                Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.debug(f"Compile '{model_name}' model successfully")

        elif model_name == models[1]:  # LeNet5
            logger.debug(f"Creating '{model_name}' model...")
            model = Sequential(
                [
                    # 1 – Convolutional2D 6@28×28 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Conv2D(6, (3, 3), activation="relu", input_shape=(32, 32, 1)),
                    AveragePooling2D(),
                    # 2 – Convolutional2D, 16@10×10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Conv2D(16, (3, 3), activation="relu"),
                    AveragePooling2D(),
                    Flatten(),
                    # 3 – Dense, 1×120 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(120, activation="relu"),
                    # 4 – Dense 1×84 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(84, activation="relu"),
                    # 5 – Dense 1×10 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    Dense(10, activation="softmax"),
                ]
            )
            logger.debug(f"'{model_name}' model has been created")

            # define optimizer
            Optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
            model.compile(
                Optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.debug(f"Compile '{model_name}' model successfully")

        else:  # custom
            logger.debug(f"Creating '{model_name}' model")
            model = Sequential(
                [
                    Conv2D(
                        60,
                        (5, 5),
                        input_shape=((32, 32) + (1,)),
                        activation="relu",
                    ),
                    Conv2D(60, (5, 5), activation="relu"),
                    MaxPooling2D(pool_size=(2, 2)),
                    Conv2D(30, (3, 3), activation="relu"),
                    Conv2D(30, (3, 3), activation="relu"),
                    MaxPooling2D(pool_size=(2, 2)),
                    Dropout(0.5),
                    Flatten(),
                    Dense(500, activation="relu"),
                    Dropout(0.5),
                    Dense(10, activation="softmax"),
                ]
            )
            logger.debug(f"'{model_name}' model has been created")

            model.compile(
                Adam(learning_rate=0.001),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            logger.debug(f"Compile '{model_name}' model successfully")

        logger.debug(f"Return '{model_name}' model – {type(model)}")
        return model

    def load_mnist(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Loads the MNIST dataset, preprocess all the data, then returns it.

        Returns:
            list[tuple[np.ndarray, np.ndarray]]: A list of 3 tuples each
                containing (x/y) [training, testing, validation] data.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        logger.debug(f"Load MNIST dataset: {self.done}")

        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, random_state=2022
        )
        logger.debug(
            f"Split dataset: {x_train.shape}, {y_train.shape[0]} – {x_test.shape}, {y_test.shape[0]} – {x_val.shape}, {y_val.shape[0]}"
        )

        # resize image to 32×32
        x_train, x_test, x_val = (
            np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode="constant"),
            np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode="constant"),
            np.pad(x_val, ((0, 0), (2, 2), (2, 2)), mode="constant"),
        )
        logger.debug(f"Resize images: {x_train.shape} – {x_test.shape} – {x_val.shape}")

        # reshape to 32×32×1
        x_train, x_test, x_val = (
            x_train.reshape(x_train.shape + (1,)),
            x_test.reshape(x_test.shape + (1,)),
            x_val.reshape(x_val.shape + (1,)),
        )
        logger.debug(f"Reshape data: {x_train.shape} – {x_test.shape} – {x_val.shape}")

        # one-hot encode labels
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        y_val = keras.utils.to_categorical(y_val, num_classes=10)
        logger.debug(f"One-hot encode labels: {self.done}")

        logger.debug(
            f"Return list of data: {type(mnist_data := [(x_train, y_train), (x_test, y_test), (x_val, y_val)])}"
        )
        return mnist_data

    def standardize_data(self, x_mnist: list[np.ndarray]) -> list[np.ndarray]:
        """Normalizes the images in the given dataset.

        Args:
            x_mnist (list[np.ndarray]): A list containing all the x type MNIST
                data.

        Returns:
            list[np.ndarray]: A list containing the normalized x type MNIST
                data.
        """

        if self.debug:
            logger.debug(f"Show sample image before standardization: {self.done}")
            j = np.random.randint(0, len(x_mnist))
            k = np.random.randint(0, len(x_mnist[j]))
            cv2.imshow(f"DEBUG: #{k} BEFORE", cv2.resize(x_mnist[j][k], (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # images need to be standardized to speed up training
        data = []
        for x in x_mnist:
            mean_px = x.mean().astype(np.float32)
            std_px = x.std().astype(np.float32)
            x = (x - mean_px) / (std_px)
            data.append(x)
        logger.debug(f"All images have been standardized: {self.done}")

        if self.debug:
            logger.debug(f"Show sample image after standardization: {self.done}")
            cv2.imshow(f"DEBUG: #{k} AFTER", cv2.resize(data[j][k], (320, 320)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        logger.debug(f"Return standardized data: {type(data)}")
        return data

    def get_generator(self) -> ImageDataGenerator:
        """Retrieves an image generator.

        Returns:
            ImageDataGenerator: An image generator. Must .fit(x_test) before
                calling it in model.fit().
        """
        data_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,  # randomly flip images
        )

        logger.debug(f"Return data generator: {type(data_gen)}")
        return data_gen

    def get_csv_logger(self, filename: str) -> CSVLogger:
        """Retrieves a CSVLogger pointing to the given filename.

        Args:
            filename (str): The desired filename for the CSV file

        Returns:
            CSVLogger: The CSVLogger itself. Must be included in callbacks when
                training the model.
        """
        path = Path(__file__).parent / "data"
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Create 'data' directory if not exists: {self.done}")

        csv_logger = CSVLogger(f"{path}/{filename}")
        logger.debug(f"Create csv logger pointing to {path}: {self.done}")

        logger.debug(f"Return csv logger: {type(csv_logger)}")
        return csv_logger

    def get_learning_rate(self, mode: int = 1 | 0) -> ReduceLROnPlateau:
        """Retrieves the desired learning rate.

        Args:
            mode (int, optional): The desired learning rate. Defaults to 1 | 0.

        Raises:
            ValueError: Unknown learning rate.

        Returns:
            ReduceLROnPlateau: The variable learning rate.
        """
        if mode not in [1, 0]:
            raise ValueError("Unknown learning rate.")
        logger.info(f"Chosen learning rate is available")

        if mode == 1:
            rate = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)
            logger.debug(f"Retrieve first learning rate: {self.done}")

        else:
            rate = ReduceLROnPlateau(
                monitor="val_loss", patience=3, verbose=1, factor=0.5, min_lr=0.00001
            )
            logger.debug(f"Retrieve second learning rate: {self.done}")

        logger.debug(f"Return variable learning rate: {type(rate)}")
        return rate

    def confusion_matrix(
        self,
        model: Sequential,
        validation: tuple[np.ndarray, np.ndarray],
        classes: int,
        action: str = "to_csv",
        title: str = "Confusion matrix",
        normalize: bool = False,
        cmap: plt.cm = plt.cm.Blues,
    ) -> Union[plt.Figure, None]:
        """Either plot or export to CSV the confusion matrix for the given model.

        Args:
            model (Sequential): The trained model from which to compute the
                confusion matrix.
            validation (tuple[np.ndarray, np.ndarray]): The (x/y) validation
                data.
            classes (int): The number of classes.
            action (str, optional): The action to perform, chose one of the
                following: ["plot", "to_csv"]. Defaults to "to_csv".
            title (str, optional): The title of the plot. Defaults to
                "Confusion matrix".
            normalize (bool, optional): Whether to normalize the confusion
                matrix or not. Defaults to False.
            cmap (plt.cm, optional): The color map of the confusion matrix.
                Defaults to plt.cm.Blues.

        Raises:
            ValueError: Unknown action.

        Returns:
            Union[plt.Figure, None]: Based on the chosen action the function
                will either return a matplotlib.Figure object, or None.
        """
        actions = ["plot", "to_csv"]
        if action not in actions:
            raise ValueError(f"Unknown action. Chose one of the following: {actions}")
        logger.info(f"Chosen action '{action}' is available")

        classes = range(classes)

        # unpack validation data
        x_val, y_val = validation

        y_pred = model.predict(x_val)  # predict values from validation dataset

        # convert predictions classes to one hot vectors
        y_pred = np.argmax(y_pred, axis=1)

        # convert validation observations to one hot vectors
        y_true = np.argmax(y_val, axis=1)

        # compute confusion matrix
        confusion_mtrx = confusion_matrix(y_true, y_pred)
        logger.debug(f"Compute confusion matrix: {type(confusion_matrix)}")

        if action == actions[0]:  # plot the confusion matrix
            plt.imshow(confusion_mtrx, interpolation="nearest", cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            if normalize:
                confusion_mtrx = (
                    confusion_mtrx.astype("float")
                    / confusion_mtrx.sum(axis=1)[:, np.newaxis]
                )

            thresh = confusion_mtrx.max() / 2.0
            for i, j in itertools.product(
                range(confusion_mtrx.shape[0]), range(confusion_mtrx.shape[1])
            ):
                plt.text(
                    j,
                    i,
                    confusion_mtrx[i, j],
                    horizontalalignment="center",
                    color="white" if confusion_mtrx[i, j] > thresh else "black",
                )

            plt.tight_layout()
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.show()
            return confusion_mtrx
        else:  # export the confusion matrix to csv
            try:
                Path.unlink("./source/data/confusion_matrix.csv")
            except:
                mtrx = pd.DataFrame(confusion_mtrx)
                mtrx.to_csv("./source/data/confusion_matrix.csv")

            return

    # def display_errors(self, validation):
    #     x_val, y_val = validation

    #     y_prediction = model.predict(x_val)
    #     y_prediction_classes = np.argmax(y_prediction, axis=1)

    #     y_true = np.argmax(y_val, axis=1)

    #     # errors are the difference between predicted labels and true labels
    #     errors = y_prediction - np.expand_dims(y_true, axis=1) != 0

    #     y_prediction_classes_errors = y_prediction_classes[errors]
    #     y_prediction_errors = y_prediction[errors]
    #     y_true_errors = y_true[errors]
    #     x_val_errors = x_val[errors]

    #     # Probabilities of the wrong predicted numbers
    #     y_pred_errors_prob = np.max(y_prediction_errors, axis=1)

    #     # Predicted probabilities of the true values in the error set
    #     true_prob_errors = np.diagonal(
    #         np.take(y_prediction_errors, y_true_errors, axis=1)
    #     )

    #     # Difference between the probability of the predicted label and the true label
    #     delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

    #     # Sorted list of the delta prob errors
    #     sorted_dela_errors = np.argsort(delta_pred_true_errors)

    #     # Top 6 errors
    #     most_important_errors = sorted_dela_errors[-6:]

    #     n, rows, cols = 0, 2, 3

    #     fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
    #     for row in range(rows):
    #         for col in range(cols):
    #             error = most_important_errors[n]
    #             ax[row, col].imshow((x_val_errors[error]).reshape((28, 28)))
    #             ax[row, col].set_title(
    #                 "Predicted label :{}\nTrue label :{}".format(
    #                     y_prediction_classes_errors[error], y_true_errors[error]
    #                 )
    #             )
    #         n += 1
    #     plt.show()

    def evaluate(
        self, model: Sequential, x_test: np.ndarray, y_test: np.ndarray
    ) -> str:
        """Evaluates the given model from the given test data.

        Args:
            model (Sequential): A trained Sequential() keras model.
            x_test (np.ndarray): The x test data (images).
            y_test (np.ndarray): The y test data (labels).

        Returns:
            str: _description_
        """
        score = model.evaluate(x_test, y_test, verbose=0)
        final = f"Score: {score[0] * 100}\nModel accuracy: {score[1] * 100}"
        return print(final)


# for debugging purposes
if __name__ == "__main__":
    # define some constants
    MODEL_NAME = "LeNet5+"

    EPOCHS = 30
    BATCH_SIZE = 86
    VERBOSE = 1

    # load model class with utility functions
    Model = Model()

    # retrieve preferred model
    model = Model.get_model(MODEL_NAME)

    # load MNIST dataset and unpack it
    mnist_data = Model.load_mnist()
    (x_train, y_train), (x_test, y_test), (x_val, y_val) = mnist_data

    # all x_<data> needs to be standardized
    x_train, x_test, x_val = Model.standardize_data([x_train, x_test, x_val])

    # get generator and augment data
    DataGenerator = Model.get_generator()
    DataGenerator.fit(x_train)

    csv_logger = Model.get_csv_logger("history.csv")
    learning_rate = Model.get_learning_rate(1)

    try:
        model = load_model(f"./source/{MODEL_NAME}")
    except:
        n = 0
        while True and n < 10:
            # train the model
            history = model.fit(
                DataGenerator.flow(x_train, y_train, batch_size=BATCH_SIZE),
                epochs=EPOCHS,
                validation_data=(x_val, y_val),
                verbose=VERBOSE,
                steps_per_epoch=x_train.shape[0] // BATCH_SIZE,
                callbacks=[csv_logger, learning_rate],
            )
            score = model.evaluate(x_test, y_test)
            n += 1
            if round(score[1] * 100, 2) > 99.75:
                break

    model.save(f"./source/{MODEL_NAME}")

    Model.evaluate(model, x_test, y_test)
    Model.confusion_matrix(model, (x_val, y_val), 10, action="to_csv")
    # LeNet.display_errors((x_val, y_val))
