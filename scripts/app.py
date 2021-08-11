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

st.set_page_config(page_title="Dashboard | Amharic Speech Recognition ", layout="wide")

st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Amharic Speech Recognition </h1>", unsafe_allow_html=True)

dynamic_range = st.sidebar.write("""
## Translate  Amharic Speech
###
""")

st.sidebar.button('Load audio file')
st.sidebar.button('Real time Translate')

st.write("""
## Select one of the following Amharic recorded Speech
""")

st.audio(path_to_wave_files)

st.write("""
## Upload Amharic audio file
""")

file = st.file_uploader("Pick a file")
