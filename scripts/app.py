# Importing packages
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

import pickle
import math
from file_handler import FileHandler
from logs import load_logging

import streamlit as st

# Setting Logs
import warnings

# Import packages for logging
import logging
import logging.handlers
import os

# Path to different Resources
PATH_TEST_WAV = "./data/AMHARIC/test/wav"
path_to_wave_files = "./data/AMHARIC/test/wav/"
MODEL_URL = "/models/model.pkl"


def load_sample_speech(audio_files_path):
    file_handler = FileHandler()
    sample_test_wav = file_handler.read_data(audio_files_path)
    sample_test_wav = [str(i)+".wav" for i in sample_test_wav]

    return sample_test_wav


def tranlate_audio(path_audio_file):
    # Dummy function
    return "This is dummy translation: ሰላም ፣ ይህ የናሙና ትርጉም ብቻ ነው ፣ እሱ እውነተኛ አይደለም"

st.set_page_config(page_title="Dashboard | Amharic Speech Recognition ", layout="wide")
st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Amharic Speech Recognition </h1>", unsafe_allow_html=True)

def main():
    load_logging()
    dynamic_range = st.sidebar.write("""
    ## Translate  Amharic Speech
    ### 
    """)

    default = "Home"
    load_file = "Load audio file"
    real_time_translation = "Real time Translate"
    transaltion_mode = st.sidebar.selectbox("Choose translation Mode", [default, load_file, real_time_translation])

    if transaltion_mode == default:
        st.success('This demo is using Deep Learning Model to process and convert African language (Amharic) speech/voice to text format.')
        st.write("""
        #### To try it out,  
        - Please load the sample Amharic speech/audio files provided,
        - Upload your Amharic audio file, or
        - Use this demo to record your Speech and translate it
        """ )

    if transaltion_mode == load_file:
        st.write("""
        ### Select one of the following Amharic recorded Speech
        """)
        try:
            samples = load_sample_speech(PATH_TEST_WAV)
            sample_audio = st.selectbox("Choose translation Mode", samples)
            st.audio(path_to_wave_files+sample_audio)
            logging.info(f" Loading sample audio file successfully")
            if st.button('Click Here to Translate'):

                # Function to translate
                
                txt = tranlate_audio("path_audio_file")
                st.text(txt)
        except Exception as e:
            logging.exception(f" Exception occured in loading sample audio file, {e}")


        st.write("""
        ### Upload Amharic audio file
        """)

        try:
            file = st.file_uploader("Pick a file")
            if file != None:
                if st.button('Click Here to Translate your Audio'):
                    # Function to translate
                    txt = tranlate_audio("path_audio_file")
                    st.text(txt)

            else:
                st.error('Please, Upload the audio file')

        except Exception as e:
            logging.exception(f" Exception occured in uploading audio file, {e}")

    elif transaltion_mode == real_time_translation:
        st.write("""
        ### Real Time Amharic Speech Recognition
         - This feature allows the user to get translations while talking.
        """)
        st.success('Comming Soon! Please Come back later')


if __name__=='__main__': 
    main()


# sudo docker container ls
# sudo docker exec -it 4f604bc51340 bash
# docker image ls
