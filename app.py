import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mask_detector.h5")

st.title("😷 Face Mask Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = img.reshape(1,224,224,3)

    pred = model.predict(img)[0][0]

    if pred < 0.5:
        st.success("Mask Detected 😷")
    else:
        st.error("No Mask 😐")