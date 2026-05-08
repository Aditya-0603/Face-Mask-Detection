import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
)

# Load model
model = tf.keras.models.load_model("mobilenet_mask_detector.h5")

st.sidebar.empty()

st.markdown("# 😷 Face Mask Detection")
st.markdown(
    "Upload a photo of a face and the model will tell you whether a mask is detected. "
    "The layout is optimized for quick image review and result display."
)

left_col, right_col = st.columns([3, 2])

with left_col:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        resized = image.resize((224, 224))
        model_input = np.array(resized) / 255.0
        model_input = model_input.reshape(1, 224, 224, 3)

        pred = float(model.predict(model_input)[0][0])
        mask_prob = 1.0 - pred
        no_mask_prob = pred

        if mask_prob > no_mask_prob:
            st.success("Mask Detected 😷")
            label = "Mask"
        else:
            st.error("No Mask Detected 😐")
            label = "No Mask"

        right_col.metric("Prediction", label, delta=f"{max(mask_prob, no_mask_prob):.2%} confidence")
        right_col.subheader("Confidence")
        right_col.write(f"- Mask: **{mask_prob:.2%}**")
        right_col.write(f"- No Mask: **{no_mask_prob:.2%}**")

        right_col.markdown("---")
        right_col.info(
            "For best results, upload an image with a clear front-facing face and good lighting."
        )
    else:
        left_col.info("Upload an image to get started.")
        right_col.write("### Example Guide")
        right_col.write(
            "- Use a JPEG or PNG image.\n"
            "- Ensure the face is centered.\n"
            "- Avoid heavy occlusion or extreme angles."
        )
      
