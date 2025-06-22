import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# Load trained model
def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1), name="input_layer"),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

model = build_model()
model.load_weights("mnist_weights.weights.h5")

st.title("Digit Classifier (MNIST)")
st.write("Upload a 28x28 image or draw your own digit below:")

# --- Option 1: Upload an image ---
uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('L').resize((28, 28))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))
    pred = model.predict(arr)
    st.image(img.resize((140, 140)), caption="Your uploaded digit")
    st.write(f"Predicted Digit: **{np.argmax(pred)}**")

st.markdown("---")

# --- Option 2: Draw your digit ---
st.subheader("Or draw a digit below (click and drag):")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Resize, convert to grayscale
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))  # take Red channel
    img_resized = img.resize((28, 28)).convert('L')
    arr = np.array(img_resized) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))

    if st.button("Predict Drawn Digit"):
        pred = model.predict(arr)
        st.image(img_resized.resize((140, 140)), caption="Your drawn digit")
        st.write(f"Predicted Digit: **{np.argmax(pred)}**")