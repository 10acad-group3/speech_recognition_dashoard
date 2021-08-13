import os
import sys
import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join('./scripts')))
from predict import Predict


def tranlate_audio(path_audio_file):
    # Dummy function
    return "This is dummy translation: ሰላም ፣ ይህ የናሙና ትርጉም ብቻ ነው ፣ እሱ እውነተኛ አይደለም"


def app():
    st.write(""" ### Upload Amharic audio file """)
    predict = Predict()

    try:
        file = st.file_uploader("Pick a file")
        if file != None:
            if st.button('Click Here to Translate your Audio'):
                # Function to translate
                txt = predict.predict(file)
                st.text(txt)

        else:
            st.error('Please, Upload the audio file')

    except Exception as e:
        print(f" Exception occurred in uploading audio file, {e}")
