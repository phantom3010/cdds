import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = tf.keras.models.load_model('crop_disease_model.h5')

# Define class names (update based on your dataset)
class_names = [
    'Apple___Black_rot', 'Apple___Healthy', 'Apple___Scab',
    'Corn___Common_rust', 'Corn___Healthy', 'Corn___Northern_Leaf_Blight',
    'Potato___Early_blight', 'Potato___Healthy', 'Potato___Late_blight',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold'
]

# Streamlit UI
st.set_page_config(page_title="Crop Disease Detector", layout="centered")
st.title("\U0001F33F Crop Disease Detection Web App")
st.write("Upload an image of a crop leaf, and the AI model will predict the disease class.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output
    st.success(f"\u2705 Predicted: **{predicted_class}** ({confidence:.2f}% confidence)")
