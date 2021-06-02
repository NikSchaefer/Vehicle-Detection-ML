import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

IMG_SIZE = 64

SAVE_PATH = "C:\\Users\\schaefer\\Desktop\\ML\\vehicle-detection\\save\\model"

IMG_PATH = "data/vehicles/1.png"
img = load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))

img = np.asarray(img)

plt.imshow(img)
plt.show()

print("\n\n", img.shape, "\n\n")

model = tf.keras.models.load_model(SAVE_PATH)

output = model.predict(img)

print(f"Predicted: {output}")
