import tensorflow as tf

# load the model
model = tf.keras.models.load_model('./inception_resnet_v2.h5')

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
