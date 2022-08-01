
# Vehicle-Detection ML

Machine Learning model built to detect if an image has a vehicle. Built with
tensorflow and keras.

## Images into data

using the Keras ImageDataGenerator the collection of the data was simple enough
with just a few lines and then moving the data into numpy arrays

```py
data_generator = ImageDataGenerator(rescale=1.0 / 255)

data = data_generator.flow_from_directory(directory="data", target_size=(64, 64))

data.reset()
x_data, y_data = next(data)

for i in range(len(data) - 1):
    img, label = next(data)
    x_data = np.append(x_data, img, axis=0)
    y_data = np.append(y_data, label, axis=0)
print("\n DATA SHAPE: ", x_data.shape, y_data.shape, "\n")
```

## Model

The Model to classify vehicle images is using the Sequential model by Keras.
This model will execute the layers added to it in Sequential order to classify
the images.

```py
model = Sequential()
```

### Layers

The Layers of the model consisted of 10 layers. The first 4 layers used 2 Conv2d
layers to sharpen and highlight features of the images. Both layers used 64
filters with a kernel size of (3, 3). Each Conv2d Layer was followed by a
MaxPooling2d Layer to bring the image back to its spatital dimensions.

The following layer is a flatten layer to prep the data for the dense layers.
The Final Dense layers dense down from 1024 to the 2 image classes(vehicle or
non-vehicle) and classify the image. The final dense layer includes a softmax
layer to bring the data back to normalization.

```py
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
```

### Optimizer

This model uses an Adam optimizer from Keras

### Loss

This model uses categorical_crossentropy loss function to penalize the model

```py
model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
```

## Accuracy

The Model was able to achieve a final test accuracy of 98.9% when evaluating the
test data.

```py
loss, acc = model.evaluate(x_test, y_test)
    print("Test Evaluation Loss: {}".format(loss))
    print("Test Evaluation Accuracy: {}".format(acc))
```

## Layout

data is split by type and then split into train and test in the code. Save
folder contains the saved model. load.py contains the loaded model from the save
folder you can run.

```py
/data
    /vehicle
    /non-vehicle
/save
```

## Data

Data is from the vehicle-detection-image-set dataset on kaggle or

`kaggle competitions download -c vehicle-detection-image-set`

with the kaggle command line


## Installation
Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
