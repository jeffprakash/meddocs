from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import BreastCancerSerializer
from .models import BreastCancerModel
from tensorflow import keras
from keras import Model
from keras.applications.inception_resnet_v2 import preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
from rest_framework import serializers
from .models import BreastCancerModel
import keras
from keras.utils import img_to_array


class BreastCancerPredictionAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = BreastCancerSerializer(data=request.data)
        if serializer.is_valid():
            # Get the image data from the serializer
            model = tf.keras.models.load_model('./inception_resnet_v2.h5', compile=False)
            #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            
        
            
            # load and preprocess the image
            image = Image.open(serializer.validated_data['image'])
            image = image.resize((299, 299))
            image = np.array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)
            
            # make the prediction
            prediction = model.predict(image)
            if prediction[0][0] > 0:
                result = "cancerous"
            else:
                result = "non-cancerous"
            
            return Response({"result": result})
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


