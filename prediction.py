from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

from PIL import Image
import tensorflow as tf
image_path = '16_left.jpeg'
image = Image.open(image_path)
model = load_model('D:\sidhu project\diabetic_retinopathy.h5')
graph = tf.compat.v1.get_default_graph()
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

processed_image = preprocess_image(image, target_size=(224, 224))
prediction = model.predict(processed_image).tolist()

print(f'prediction that the person has Diabetic Retinopathy {prediction[0][0]*100}%')
print(f'prediction that the person has no Diabetic Retinopathy {prediction[0][1]*100}%')

