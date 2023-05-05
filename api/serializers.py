import numpy as np
from rest_framework import serializers
from .models import BreastCancerModel
import keras
from keras.utils import img_to_array





class BreastCancerSerializer(serializers.Serializer):
    image = serializers.ImageField()
    
    def validate_image(self, value):
        # Check if the uploaded file is an image
        if not value.content_type.startswith('image'):
            raise serializers.ValidationError('Please upload an image file.')
        return value

    def predict_disease(self, image):
        # Load the model weights
        model = BreastCancerModel()
        model.load_weights('./inception_resnet_v2.h5', compile=False)


        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        
        # Preprocess the image for model input
        img = keras.preprocessing.image.load_img(image, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = keras.applications.inception_resnet_v2.preprocess_input(x)
        
        # Make the prediction
        y_pred = model.predict(x)
        return y_pred[0][0]
    
    def create(self, validated_data):
        # Get the image file from validated data
        image = validated_data.get('image', None)
        
        # Predict the disease using the image
        disease_prediction = self.predict_disease(image)
        
        # Return the prediction as a dictionary
        return {'disease_prediction': disease_prediction}
