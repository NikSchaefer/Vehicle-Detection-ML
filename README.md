# Vehicle-Detection ML
Machine Learning model built to detect if an image has a vehicle. Built with tensorflow and keras.

## Layers
The model has 2 Convolational 2d Layers followed by MaxPooling layers to highlight features. The flatten layer flattens the data and then the following dense layers dense down to a decision if the image has a vehicle. 

## Accuracy
The Model achieved 98.9 accuracy when evaluating the model on the test set.

## Layout

data is split by type and then split into train and test in the code. Save folder contains the saved model. `load.py` contains the loaded model from the save folder you can run.

```
/data
    /vehicle
    /non-vehicle
/save
```

## Data
Data is from the vehicle-detection-image-set dataset on [kaggle](https://www.kaggle.com/brsdincer/vehicle-detection-image-set) or `kaggle competitions download -c vehicle-detection-image-set` with the kaggle command line

## Installation
Install Python 3.8+

```bash
pip install tensorflow keras numpy
```

## License
[MIT](https://choosealicense.com/licenses/mit/)