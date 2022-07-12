import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D


# TODO: complete refactor

# constants ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH = "dataset"
# PATH_LABELS = "labels.csv"

# ratios+
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
IMG_DIMS = (32, 32)

# model
BATCH_SIZE = 50
EPOCHS = 10
STEPS = 131
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# list directories inside PATH
def get_classes(path: str) -> list:
    __dir_list = os.listdir(PATH)
    print(f"Total num of classes detected: {len(__dir_list)}")  # should be 10

    # store the number of folders. the latter is also the number of classes
    __classes = len(__dir_list)
    return __classes


# import all images and insert them inside a single list
def import_classes(__classes: list) -> list:
    print("Importing classes...")

    images, class_ids = [], []
    # loop folders to obtain images inside them
    for _class in range(__classes):
        __img_list = os.listdir(f"{PATH}/{str(_class)}")

        # for every folder loop through the images
        for _img in __img_list:
            current_img = cv2.imread(f"{PATH}/{str(_class)}/{str(_img)}")

            # images are too big to be processed efficiently by the network thus
            # a resize is needed (180×180 -> 32×32)
            current_img = cv2.resize(current_img, IMG_DIMS)
            images.append(current_img)

            # save the corresponding ID (class ID) of each image
            class_ids.append(_class)
        print(_class, end=" ")
    print()  # reset previous end command

    return images, class_ids


# convert lists to numpy arrays
def to_array(images: list, class_ids: list) -> np.ndarray:
    images, class_ids = np.array(images), np.array(class_ids)
    return images, class_ids


# the images.shape is (10160, 32, 32, 3). It's list a 32×32 images with 3 colors
# channels (RGB). The class_ids only has the firs number

# split dataset to create sets fot testing, training and validation, using
# train_test_split() from sklearn.model_selection
def split_data(images: list, class_ids: list, ratio: float) -> np.ndarray:
    X_train, X_test, Y_train, Y_test = train_test_split(
        images, class_ids, test_size=ratio
    )


# __classes = get_classes(PATH)  # generate classes
# images, class_ids = import_classes(__classes)  # get lists

# images, class_ids = to_array(images, class_ids)  # convert lists

# now is very easy to check the shape
# print(f"All imgs: {imgs.shape}\nAll class: {class_num.shape}")

# # Splitting the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# x_train, x_test, y_train, y_test = train_test_split(
#     imgs, class_num, test_size=TEST_RATIO
# )
# # training = 0.8, testing = 0.2

# # we need to split the data into training and testing because we need to shuffle
# # all the classes. the first 80% will be till 7 (folder) and will now include 8,
# # 9. this is why we need a package to suffle data and split it evenly to train
# # and test accordingly

# x_train, x_validation, y_train, y_validation = train_test_split(
#     x_train, y_train, test_size=VALIDATION_RATIO
# )

# print(f"Train test: {x_train.shape}")
# print(f"Test set: {x_test.shape}")
# print(f"Validation set: {x_validation.shape}")

# # since we do not want to favour any class by having more data from one
# # specific class. we can get this info from the y_train.
# # x_train contains the actual images and y_train contains the IDs of each images

# num_of_samples = [len(np.where(y_train == i)[0]) for i in range(0, num_of_classes)]
# print(f"number fo samples (for num): {num_of_samples}")

# # visualize the data with a bar chart
# plt.figure(figsize=(10, 5))
# plt.bar(range(0, num_of_classes), num_of_samples)
# plt.title("number of images for each class")
# plt.xlabel("Class ID")
# plt.ylabel("number of images")
# # plt.show()


# # the next step is to pre-process all the images
# def preprocess(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
#     img = cv2.equalizeHist(img)  # equalize img (distribute lighting evenly)
#     img = img / 255  # normalize img (restrict value from [0, 255] to [0, 1])

#     return img


# # just as a check
# # img = preprocess(x_train[31])
# # img = cv2.resize(img, (300, 300))  # resize img since start img is small
# # cv2.imshow("preprocessed", img)
# # cv2.waitKey(0)  # wait

# # we need to preprocess all the images. we can use map() we need to overwrite
# # the lists with the newly created list (array)
# x_train = np.array(list(map(preprocess, x_train)))
# x_test = np.array(list(map(preprocess, x_test)))
# x_validation = np.array(list(map(preprocess, x_validation)))

# # add depth of one to imgs. required for CNN to run properly
# # ci sarà un metodo meno brutto
# # infatti c'è
# # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# # x_validation = x_validation.reshape(
# # x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1
# # )
# x_train = x_train.reshape(x_train.shape + (1,))
# x_test = x_test.reshape(x_test.shape + (1,))
# x_validation = x_validation.reshape(x_validation.shape + (1,))


# # next we need to augment the images (zoom, rotation, translation), makes
# # dataset more generic
# DataGen = ImageDataGenerator(
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.2,
#     shear_range=0.1,
#     rotation_range=0.1,
# )

# DataGen.fit(x_train)

# # next step is to one hot encode the matrices
# y_train = to_categorical(y_train, num_of_classes)
# y_test = to_categorical(y_test, num_of_classes)
# y_validation = to_categorical(y_validation, num_of_classes)


# # create the model
# def model():
#     FILTERS = 60
#     FILTER1 = (5, 5)
#     FILTER2 = (3, 3)
#     POOL_SIZE = (2, 2)
#     NODES = 500

#     model = Sequential()
#     model.add(
#         Conv2D(
#             FILTERS,
#             FILTER1,
#             input_shape=(IMG_DIMS[0], IMG_DIMS[1], 1),
#             activation="relu",
#         )
#     )
#     model.add(Conv2D(FILTERS, FILTER1, activation="relu"))

#     # add pooling layer
#     model.add(MaxPooling2D(pool_size=POOL_SIZE))

#     # add two additional convolutional layers halving the number of filters and
#     # changing to filter 2
#     model.add(Conv2D(FILTERS // 2, FILTER2, activation="relu"))
#     model.add(Conv2D(FILTERS // 2, FILTER2, activation="relu"))

#     model.add(MaxPooling2D(pool_size=POOL_SIZE))

#     model.add(Dropout(0.5))  # 50%

#     # add Flatten layer
#     model.add(Flatten())

#     # add dense layer
#     model.add(Dense(NODES, activation="relu"))
#     model.add(Dropout(0.5))  # 50%
#     model.add(Dense(num_of_classes, activation="softmax"))

#     model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

#     return model


# # assign and get summary of model
# model = model()
# print(model.summary())

# history = model.fit(
#     DataGen.flow(x_train, y_train, batch_size=BATCH_SIZE),
#     steps_per_epoch=STEPS,
#     epochs=EPOCHS,
#     validation_data=(x_validation, y_validation),
#     shuffle=1,
# )

# plt.figure(1)
# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.legend(["training", "validation"])
# plt.title("Loss")
# plt.xlabel("Number of epochs")
# # plt.show()

# plt.figure(2)
# plt.plot(history.history["accuracy"])
# plt.plot(history.history["val_accuracy"])
# plt.legend(["training", "validation"])
# plt.title("Accuracy")
# plt.xlabel("Number of epochs")

# plt.show()

# score = model.evaluate(x_test, y_test, verbose=0)
# print(f"Score: {score[0]}\nAccuracy: {score[1]}")


# # pickle_output = open("model_trained.h5", "wb")
# # pickle.dump(model, pickle_output)
# # pickle_output.close()

# model.save("trained_model.h5")
