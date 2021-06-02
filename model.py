from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


TRAIN_END = 14000
TEST_START = TRAIN_END + 1

EPOCHS = 5
BATCH_SIZE = 32

SAVE_PATH = "save/model"


def split_for_test(list):
    train = list[0:TRAIN_END]
    test = list[TEST_START:]
    return train, test


def main():
    data_generator = ImageDataGenerator(rescale=1.0 / 255)

    data = data_generator.flow_from_directory(directory="data", target_size=(64, 64))

    data.reset()
    x_data, y_data = next(data)

    for i in range(len(data) - 1):
        img, label = next(data)
        x_data = np.append(x_data, img, axis=0)
        y_data = np.append(y_data, label, axis=0)
    print("\n DATA SHAPE: ", x_data.shape, y_data.shape, "\n")

    x_train, x_test = split_for_test(x_data)
    y_train, y_test = split_for_test(y_data)

    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(64, 64, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    loss, acc = model.evaluate(x_test, y_test)
    print("Test Evaluation Loss: {}".format(loss))
    print("Test Evaluation Accuracy: {}".format(acc))

    model.save(SAVE_PATH)

"""
Accuracy:
0.9893588423728943
"""

if __name__ == "__main__":
    main()
