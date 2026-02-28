import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np

# Load the ResNet50 model with pre-trained ImageNet weights
model = ResNet50(weights='imagenet')

# Load and preprocess an example image
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # Preprocess input for the ResNet model

# Predict the class of the image
predictions = model.predict(x)

# Decode and print the top 3 predictions
print('Predicted:', decode_predictions(predictions, top=3)[0])
