from keras.applications.inception_resnet_v2 import InceptionResNetV2

# Create an instance of the InceptionResNetV2 model
model = InceptionResNetV2(weights='imagenet')

# Save the model weights to a file
model.save('inception_resnet_v2.h5')
