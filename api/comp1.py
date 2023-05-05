import tensorflow as tf
import numpy as np
from rest_framework import serializers
import keras
from keras.utils import img_to_array



# load the model
model = tf.keras.models.load_model('./inception_resnet_v2.h5')

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
