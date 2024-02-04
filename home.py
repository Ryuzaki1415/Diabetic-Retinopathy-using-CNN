
import streamlit as st
st.title('Diabetic retinopathy ')

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

# returns a compiled model
# identical to the previous one

from PIL import Image
import tensorflow as tf
try:
    uploaded_file = st.file_uploader("Choose a file")
except:
    st.write('please upload a valid image')
if uploaded_file is not None:
    st.write("Uploaded Image")
    st.image(uploaded_file)
    image = Image.open(uploaded_file)
    model = load_model('D:\sidhu project\diabetic_retinopathy.h5')
    graph = tf.compat.v1.get_default_graph()
    def preprocess_image(image, target_size):
        st.write("preprocessing image....")
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    processed_image = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(processed_image).tolist()

    st.write("MODEL PREDICTION")
    st.write(f'prediction that the person has Diabetic Retinopathy {prediction[0][0]*100}%')
    st.write(f'prediction that the person has no Diabetic Retinopathy {prediction[0][1]*100}%')
    
