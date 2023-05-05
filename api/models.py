from django.db import models
from django.db import models
from tensorflow import keras
from keras import Model
from keras.applications import InceptionResNetV2
from keras.layers import Dense, Dropout


class Image(models.Model):
    image = models.ImageField(upload_to='images/')


class BreastCancerModel(Model):
    def __init__(self):
        super().__init__()
        self.base_model = InceptionResNetV2(
            include_top=False,
            weights=None,
            input_shape=(299, 299, 3),
            pooling='avg'
        )
        self.dropout = Dropout(0.5)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x


# Load the saved Keras model
# model = keras.models.load_model('inception_resnet_v2.h5')

# Compile the model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a serializer for the BreastCancer model

# Define a viewset for the BreastCancer model
# class BreastCancerViewSet(viewsets.ModelViewSet):
#     serializer_class = BreastCancerSerializer

#     def create(self, request):
#         serializer = self.serializer_class(data=request.data)
#         if serializer.is_valid():
            # Load the image and preprocess it for the Keras model
            # img = keras.preprocessing.image.load_img(serializer.validated_data['image'], target_size=(299, 299))
            # x = keras.preprocessing.image.img_to_array(img)
            # x = keras.applications.inception_resnet_v2.preprocess_input(x)
            # x = np.expand_dims(x, axis=0)

            # Make a prediction with the Keras model
            # result = model.predict(x)[0][0]

            # Save the BreastCancer object with the prediction result
            # breast_cancer = serializer.save(result=result)

            # return Response(serializer.data, status=status.HTTP_201_CREATED)

        # return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
