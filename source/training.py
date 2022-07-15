#!/usr/bin/env python3

import utils as u

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FILENAME = "dataset"
PARENT_PATH = u.parent_path()

TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2

# model
FILTERS = 60
FILTER1 = (5, 5)
FILTER2 = (3, 3)
POOL_SIZE = (2, 2)
NODES = 500

# training
BATCH_SIZE = 50
EPOCHS = 15
STEPS = 131
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def main() -> None:
    dr = u.DataReader(FILENAME)
    images, class_ids = dr.import_data()

    # divide the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        images, class_ids, test_size=TEST_RATIO
    )

    # divide again
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train, test_size=VALIDATION_RATIO
    )

    # pre-process images
    pre = u.PreProcessor(X_train, X_test, X_validation)
    X_train, X_test, X_validation = pre.process_data()

    # augment images
    DataGenerator = u.augment_images(X_train)

    # one-hot encode matrices
    classes = dr.get_classes()  # retrieve classes
    Y_train, Y_test, Y_validation = (
        to_categorical(Y_train, classes),
        to_categorical(Y_test, classes),
        to_categorical(Y_validation, classes),
    )

    # get model
    model = u.get_model(FILTERS, FILTER2, FILTER2, POOL_SIZE, NODES, classes)

    # create a csv logger
    logger = u.model_logger("model.csv")

    # print summary
    # print(model.summary())

    # train the model
    history = model.fit(
        DataGenerator.flow(X_train, Y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=STEPS,
        epochs=EPOCHS,
        validation_data=(X_validation, Y_validation),
        shuffle=1,
        callbacks=[logger],
    )

    # save the model
    model.save(f"{PARENT_PATH}/trained_model")


if __name__ == "__main__":
    main()
