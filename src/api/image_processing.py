import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def preprocess_image(image):
    image = image.convert('L').resize((48, 48))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
