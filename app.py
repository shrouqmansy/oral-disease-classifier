import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = "best_model.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def predict(model, image):
    image = ImageOps.exif_transpose(image).convert("RGB").resize(IMG_SIZE)
    # raw 0-255 pixels: EfficientNet has its own preprocessing inside the model
    x = np.expand_dims(np.array(image, dtype="float32"), axis=0)
    return model.predict(x, verbose=0)[0]


st.title("🦷 Oral Disease Classification")
st.write("Upload a photo of the mouth/teeth and the model will predict the condition.")

model, class_names = load_model()

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        probs = predict(model, image)

    top = int(np.argmax(probs))
    st.subheader(f"Prediction: {class_names[top].replace('_', ' ')}")
    st.metric("Confidence", f"{probs[top] * 100:.1f}%")
    if probs[top] < 0.6:
        st.warning("Low confidence - the result may be unreliable.")

    st.write("All class probabilities:")
    st.bar_chart({class_names[i].replace("_", " "): float(p) for i, p in enumerate(probs)})

st.caption("For educational purposes only - not a medical diagnosis.")
