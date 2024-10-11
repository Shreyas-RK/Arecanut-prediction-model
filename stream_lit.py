# from sklearn.base import BaseEstimator, ClassifierMixin
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import streamlit as st
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# # class KerasClassifier(BaseEstimator, ClassifierMixin):
# #     def _init_(self, model):
# #         self.model = model

# #     def fit(self, X, y):
# #         # Keras models don't need to be 'fit' in this context; we only need to predict
# #         return self
    
# #     def predict(self, X):
# #         preds = self.model.predict(X)
# #         return np.argmax(preds, axis=1)


# def create_model():
#     return loaded_model  # Assuming `loaded_model` is a pre-trained Keras model

# classifier = KerasClassifier(build_fn=create_model)


# # Load your model
# loaded_model = keras.models.load_model('my_train_model.h5')
# classifier = KerasClassifier(loaded_model)

# def preprocess_image(img_path):
#     img = image.load_img(img_path, target_size=(32, 32))
#     img_array = image.img_to_array(img)
#     normalized_image = img_array / 255.0
#     return np.expand_dims(normalized_image, axis=0)


# def plot_confusion_matrix(true_labels, predicted_labels, class_names):
#     cm = confusion_matrix(true_labels, predicted_labels)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
#     plt.show()

# st.title("Image Classifier")

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     img_path = uploaded_file.name
#     with open(img_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#     st.write("")
    
#     if st.button("Predict"):
#         processed_image = preprocess_image(img_path)
#         prediction = classifier.predict(processed_image)
#         predicted_class = class_names[prediction[0]]
#         st.write(f"The predicted class is: {predicted_class}")

import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

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