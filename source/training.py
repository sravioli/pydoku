import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# the MNIST data is split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Reshape to be samples*pixels*width*height
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype("float32")

# One hot encode
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# convert from integers to floats
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

# normalize to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0


model = Sequential()
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        input_shape=(28, 28, 1),
    )
)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"))
model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
model.add(Dense(10, activation="softmax"))
# model.summary()


# compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.save("./source/model.h5")
print("Saved model to disk")
