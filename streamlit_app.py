import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import ntpath

# Get the full path to the model file
model_path = os.path.join(os.getcwd(), 'model/classification_model.h5')
model_path.replace(os.sep,ntpath.sep)

print(model_path)

# model_path = 'c:\\Users\\SIDHARTH\\Documents\\maths_mini_project\\animal_classification_model.h5'
# Load the saved model
model = tf.keras.models.load_model(model_path)

# Define the labels for prediction
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

# Configure Streamlit layout
st.set_page_config(page_title="Object Classification", layout="wide")

# Custom CSS to enhance the appearance
st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #F63366;
            color: #FFFFFF;
            font-weight: bold;
            padding: 0.75rem 2rem;
            border-radius: 0.25rem;
            border: none;
            box-shadow: none;
        }
        .st-file-uploader {
            padding: 1rem 0;
        }
        .st-file-uploader>div>div:first-child {
            width: 100%;
        }
        .stImage>img {
            object-fit: contain;
            max-width: 100%;
            max-height: 500px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main layout
col1, col2 = st.columns([1, 3])

# Left panel - Model capabilities
with col1:
    st.image("Images/image.png", use_column_width=True)
    st.title("Model Capabilities")
    st.write("This model can predict the following:")
    st.write(class_labels)

# Right panel - Image upload and prediction
with col2:
    st.title("Object Classification")
    st.write("Upload an image and get a prediction!")

    # Image upload section
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Perform prediction on the uploaded image
    if uploaded_image is not None:
        # Read the image
        img = image.load_img(uploaded_image, target_size=(32, 32))
        img_ss = image.load_img(uploaded_image, target_size=(256, 256))
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image
        img_array = img_array / 255.0

        # Make prediction
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)

        # Display the predicted label
        st.write(f"Predicted label: {class_labels[predicted_label]}")
        st.image(img_ss, caption="Uploaded Image", use_column_width=False)
