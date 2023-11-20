import streamlit as st
import cv2
import zipfile
import tempfile
import os
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from streamlit_webrtc import webrtc_streamer


import av


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:]

    return av.VideoFrame.from_ndarray( format="bgr24")

def audio_frame_callback(frame):
    audio = frame.to_ndarray()
    return av.AudioFrame.from_ndarray(audio, format="s16")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback, audio_frame_callback=audio_frame_callback)


# Capture the webcam stream
cap = cv2.VideoCapture(0)
stream = 'fermodel.h5.zip'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if stream is not None:
    myzipfile = zipfile.ZipFile(stream)
    with tempfile.TemporaryDirectory() as tmp_dir:
        myzipfile.extractall(tmp_dir)
        root_folder = myzipfile.namelist()[0]
        model_dir = os.path.join(tmp_dir, root_folder)
        model = tf.keras.models.load_model(model_dir)
        st.info('Model loaded')
        st.write(model.summary())

status_container = st.empty()

while cap is not None:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    for (x, y, w, h) in faces:
        cropped_face = frame[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (48, 48))
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        cropped_face = cropped_face.reshape(1, 48, 48, 1)
        # st.image(cropped_face)
        pred = model.predict(cropped_face)

        

        #DISPLAY THE RESULT

        if pred[0][0] > pred[0][1]:
            stat = 'Happy'
        if pred[0][1] > pred[0][0]:
            stat = 'Neutral'

        if pred[0][2] > pred[0][1]:
            stat = 'Angry'

        status_container.empty()
        status_container.write(stat)
   

cap.release()
