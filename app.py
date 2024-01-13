import streamlit as st
from PIL import Image
import os
from keras.models import load_model
import numpy as np
import hydralit_components as hc



# Load the pre-trained Keras model

model_path = "keras_model.h5"
model = load_model(model_path, compile=False)

# Define your list of class names
class_names = ["pitutary", "notumour", "meninguama", "glioma" ]  # Add the actual names based on your model

st.title('brain tumour detection')

img_file = st.file_uploader('Upload mri image', type=['png', 'jpg', 'jpeg'])

def load_img(img):
    img = Image.open(img)
    return img

def preprocess_image(image):
    # Resize and preprocess the image for the Keras model
    image = image.resize((224, 224))
    image_array = np.asarray(image)
    
    # Ensure the input has three channels (for RGB images)
    if len(image_array.shape) == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

if img_file is not None:
    file_details = {}
    file_details['name'] = img_file.name
    file_details['size'] = img_file.size
    file_details['type'] = img_file.type
    st.write(file_details)
    st.image(load_img(img_file), width=255)

    with open(os.path.join('uploads', 'src.jpg'), 'wb') as f:
        f.write(img_file.getbuffer())

    st.success('Image Saved')

    # Load and preprocess the image for the Keras model
    image = load_img('uploads/src.jpg')
    processed_image = preprocess_image(image)

    # Predict using the Keras model
    prediction = model.predict(processed_image)
    # Assuming your model outputs probabilities for each class, adjust this part based on your model architecture
    predicted_class_index = np.argmax(prediction)
    
    # Display the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    st.success(f'Predicted Class Name: {predicted_class_name}')
