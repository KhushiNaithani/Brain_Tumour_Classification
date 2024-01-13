import pickle
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

with open("brain_tumour.pkl",'rb' ) as file:
    CNN=pickle.load(file)
    
def predict(image):
    img = Image.open(image) 
    img = img.resize((224,224)) 
    img_array =tf.keras.preprocessing.image.img_to_array(img)
    img_Array= tf.expand_dims(img_array,0)
    prediction=CNN.predict(img_Array)
    return prediction
def map_prediction(prediction):
    cl_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    return cl_labels[np.argmax(prediction)]


st.title("Brain Tumour Classification")
image_file = st.file_uploader(label="user_image_uploder",type=['jpeg','png'])
if image_file:
    st.image(image_file)
    prediction_output = predict(image_file)
    f'Class whose max ratio: {map_prediction(prediction_output)}'



