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
MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"


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

        st.audio(path_to_wave_files)
        if st.button('Translate'):
            # Function to translate
            st.text_area('Hit Translate to get Amharic translations texts.. ')

        st.write("""
        ### Upload Amharic audio file
        """)

        file = st.file_uploader("Pick a file")
        if st.button('Translate button'):
            # Function to translate

            st.text_area('Hit Translate to get Amharic translations texts')

    elif transaltion_mode == real_time_translation:
        pass


if __name__=='__main__': 
    main()