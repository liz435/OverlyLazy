import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

st.title('Hello, world!')

stream = 'fermodel.h5.zip'
img = 'test.png'

import zipfile
import tempfile

if stream is not None:
    myzipfile = zipfile.ZipFile(stream)
    with tempfile.TemporaryDirectory() as tmp_dir:
        myzipfile.extractall(tmp_dir)
        root_folder = myzipfile.namelist()[0]
        model_dir = os.path.join(tmp_dir, root_folder)
        model = tf.keras.models.load_model(model_dir)
        st.info('Model loaded')
        st.write(model.summary())

# img = st.file_uploader('some image', type=['png', 'jpg', 'jpeg'])
if img is not None:
    try:
        img = Image.open(img)
        st.write(f"Image object: {img}")
        if img is not None:
            # Convert RGBA to RGB
            img = img.convert('RGB')
            img = np.array(img)
            st.write(f"Image array shape: {img.shape}")
            
            # Check if the image has the correct shape
            if img.shape[:2] != (48, 48):
                img = img.resize((48, 48))
                st.write("Resized image to (48, 48)")

            # Check if the image has the correct type
            st.write(f"Image array dtype: {type(img)}")
            if img.dtype != 'float32':
                img = img.astype('float32')
                st.write("Converted image to float32")

            img = img / 255.0
            img = img.reshape(1, 48, 48, 3)
            
            pred = model.predict(img)
            st.write(pred)
        else:
            st.write('Failed to load image. Please check the file path and format.')
    except Exception as e:
        st.write(f"Error processing image: {str(e)}")
else:
    st.write('No image selected.')
