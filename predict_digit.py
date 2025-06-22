import tensorflow as tf
import sys, numpy as np

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

model = tf.keras.models.load_model('mnist_cnn.keras')

img = tf.keras.preprocessing.image.load_img(sys.argv[1], color_mode='grayscale', target_size=(28,28))
arr = np.expand_dims(np.array(img)/255.0, axis=(0,-1))

pred = model.predict(arr)
print("Predicted digit:", np.argmax(pred))

import matplotlib.pyplot as plt

plt.imshow(img, cmap='gray')
plt.title(f"Predicted Digit: {np.argmax(pred)}")
plt.axis('off')
plt.show()