

import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# Load your model
model = keras.models.load_model('my_train_model.h5')

# Define the class names based on your model's training
class_names = ['Jam', 'Jini', 'Lindi', 'Mora', 'Mothi', 'Sevardhana']

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((32, 32))  # Resize to the input size of your model
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)

# Streamlit UI
st.title("ARECANUT PREDICTION MODEL")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file)  # Load the image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Predict"):
        processed_image = preprocess_image(img)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_names[predicted_class_index]
        
        st.write(f"The predicted class is: {predicted_class_label}")