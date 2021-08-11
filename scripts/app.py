# Importing packages
import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px

import altair as alt
import plotly.express as px
import pickle
import math

import streamlit as st

# Setting Logs
import warnings

# Import packages for logging
import logging
import logging.handlers

path_to_wave_files = "../speech_recognition/data/AMHARIC/test/wav/01_d501021.wav"
MODEL_URL = "https://github.com/10acad-group3/speech_recognition/tree/main/models/model.pkl"

def tranlate_audio(path_audio_file):
    # Dummy function
    return "This is dummy translation: ሰላም ፣ ይህ የናሙና ትርጉም ብቻ ነው ፣ እሱ እውነተኛ አይደለም"

st.set_page_config(page_title="Dashboard | Amharic Speech Recognition ", layout="wide")
st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Amharic Speech Recognition </h1>", unsafe_allow_html=True)

def main():
    
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
            st.audio(path_to_wave_files)
            if st.button('Click Here to Translate'):
                # Function to translate
                txt = tranlate_audio("path_audio_file")
                st.text(txt)

        except Exception as e:
            pass


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
            pass

    elif transaltion_mode == real_time_translation:
        pass


if __name__=='__main__': 
    main()