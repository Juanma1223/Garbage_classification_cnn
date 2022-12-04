import numpy as np
import os
# import PIL
# import PIL.Image
import tensorflow as tf
# import tensorflow_datasets as tfds
import pathlib
from tensorflow.keras import datasets, layers, models

dataDir = "./archive/garbage_classification/Garbage_classification"

imgHeight = 384
imgWidth = 512
batchSize = 32

trainDs = tf.keras.utils.image_dataset_from_directory(
  dataDir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(imgHeight, imgWidth),
  batch_size=batchSize)

valDs = tf.keras.utils.image_dataset_from_directory(
  dataDir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(imgHeight, imgWidth),
  batch_size=batchSize)

AUTOTUNE = tf.data.AUTOTUNE

trainDs = trainDs.cache().prefetch(buffer_size=AUTOTUNE)
valDs = valDs.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(6)
])



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  trainDs,
  validation_data=valDs,
  epochs=10
)
