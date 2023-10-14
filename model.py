import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
from io import BytesIO

# Load the trained model
model = keras.models.load_model('baseline_model.h5')

# Define a function to preprocess the input image
def preprocess_image(image):
    # Preprocess the image according to your training data preprocessing steps
    # For example, resize it to (224, 224) and convert to grayscale if needed
    # You may also need to normalize the pixel values
    processed_image = image.resize((224, 224)).convert('L')
    processed_image = np.array(processed_image) / 255.0  # Normalize pixel values
    return processed_image

st.title("Your Streamlit Model Deployment")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image:
    img = Image.open(uploaded_image)
    preprocessed_image = preprocess_image(img)
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    class_id = np.argmax(prediction)
    class_name = "Your_Class_Labels[class_id]"
    confidence = float(prediction[0][class_id])

    st.image(img, caption=f"Uploaded Image", use_column_width=True)
    st.write(f"Prediction: {class_name} with confidence: {confidence}")


