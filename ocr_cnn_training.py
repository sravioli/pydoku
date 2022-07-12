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

# settings ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PATH = "dataset"
PATH_LABELS = "labels.csv"
TEST_RATIO = 0.2
VALIDATION_RATIO = 0.2
IMG_DIMS = (32, 32, 3)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
imgs = []
class_num = []

# we need the names of each one of the folders, we need to know how many
# folders there are and their names
dir_list = os.listdir(PATH)
# check
print(f"Total num of classes detected: {len(dir_list)}")  # 10

# 10 folders means 10 different classes. we can store it
num_of_classes = len(dir_list)


# next import all images and put them in a list
print("Importing classes...")
for x in range(num_of_classes):
    img_list = os.listdir(PATH + "/" + str(x))

    # for every folder we need to read the img and put it in a list
    for y in img_list:
        current_img = cv2.imread(PATH + "/" + str(x) + "/" + str(y))

        # imgs are 180×180 and it's too large for network. 32×32 is preferred
        current_img = cv2.resize(current_img, (IMG_DIMS[0], IMG_DIMS[1]))
        imgs.append(current_img)

        # we next have to save the corresponding ID (class ID) of each of these
        # imgs when x = 0, class_num = 0
        class_num.append(x)
    print(x, end=" ")  # check
print("")  # reset end=" "

# we can check how many images we actually import
print(len(imgs))  # 10160
print(len(class_num))  # same as above


# we now have to convert the lists into a numpy array
imgs = np.array(imgs)
class_num = np.array(class_num)

# now is very easy to check the shape
print(f"All imgs: {imgs.shape}\n" f"All class: {class_num.shape}")
# we have 32x32 of three channels (so it's colored) and of these images whe
# have 10160 in one matrix. next matrix has all the class numbers (matches the
# first matrix).

# to split the data we create a set of testing, training and validation
# we use sklearn.model_selection to do so

# Splitting the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_train, x_test, y_train, y_test = train_test_split(
    imgs, class_num, test_size=TEST_RATIO
)
# training = 0.8, testing = 0.2

# we need to split the data into training and testing because we need to shuffle
# all the classes. the first 80% will be till 7 (folder) and will now include 8,
# 9. this is why we need a package to suffle data and split it evenly to train
# and test accordingly

x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=VALIDATION_RATIO
)

print(f"Train test: {x_train.shape}")
print(f"Test set: {x_test.shape}")
print(f"Validation set: {x_validation.shape}")

# since we do not want to favour any class by having more data from one
# specific class. we can get this info from the y_train.
# x_train contains the actual images and y_train contains the IDs of each images

num_of_samples = [len(np.where(y_train == i)[0]) for i in range(0, num_of_classes)]
print(f"number fo samples (for num): {num_of_samples}")

# visualize the data with a bar chart
plt.figure(figsize=(10, 5))
plt.bar(range(0, num_of_classes), num_of_samples)
plt.title("number of images for each class")
plt.xlabel("Class ID")
plt.ylabel("number of images")
# plt.show()


# the next step is to pre-process all the images
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale
    img = cv2.equalizeHist(img)  # equalize img (distribute lighting evenly)
    img = img / 255  # normalize img (restrict value from [0, 255] to [0, 1])

    return img


# just as a check
# img = preprocess(x_train[31])
# img = cv2.resize(img, (300, 300))  # resize img since start img is small
# cv2.imshow("preprocessed", img)
# cv2.waitKey(0)  # wait

# we need to preprocess all the images. we can use map() we need to overwrite
# the lists with the newly created list (array)
x_train = np.array(list(map(preprocess, x_train)))
x_test = np.array(list(map(preprocess, x_test)))
x_validation = np.array(list(map(preprocess, x_validation)))

# add depth of one to imgs. required for CNN to run properly
# ci sarà un metodo meno brutto
# infatti c'è
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_validation = x_validation.reshape(
# x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1
# )
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
x_validation = x_validation.reshape(x_validation.shape + (1,))


# next we need to augment the images (zoom, rotation, translation), makes
# dataset more generic
DataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=0.1,
)

DataGen.fit(x_train)

# next step is to one hot encode the matrices
y_train = to_categorical(y_train, num_of_classes)
y_test = to_categorical(y_test, num_of_classes)
y_validation = to_categorical(y_validation, num_of_classes)


# create the model
def model():
    FILTERS = 60
    FILTER1 = (5, 5)
    FILTER2 = (3, 3)
    POOL_SIZE = (2, 2)
    NODES = 500

    model = Sequential()
    model.add(
        Conv2D(
            FILTERS,
            FILTER1,
            input_shape=(IMG_DIMS[0], IMG_DIMS[1], 1),
            activation="relu",
        )
    )
    model.add(Conv2D(FILTERS, FILTER1, activation="relu"))

    # add pooling layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    # add two additional convolutional layers halving the number of filters and
    # changing to filter 2
    model.add(Conv2D(FILTERS // 2, FILTER2, activation="relu"))
    model.add(Conv2D(FILTERS // 2, FILTER2, activation="relu"))

    model.add(MaxPooling2D(pool_size=POOL_SIZE))

    model.add(Dropout(0.5))  # 50%

    # add Flatten layer
    model.add(Flatten())

    # add dense layer
    model.add(Dense(NODES, activation="relu"))
    model.add(Dropout(0.5))  # 50%
    model.add(Dense(num_of_classes, activation="softmax"))

    model.compile(Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    return model


# assign and get summary of model
model = model()
print(model.summary())

BATCH_SIZE = 50
EPOCHS = 10
STEPS = 20_000

history = model.fit(
    DataGen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=STEPS,
    epochs=EPOCHS,
    validation_data=(x_validation, y_validation),
    shuffle=1,
)
