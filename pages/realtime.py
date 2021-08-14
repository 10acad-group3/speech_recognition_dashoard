import os
import sys
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import pickle
import wavio
import sounddevice as sd
sys.path.append(os.path.abspath(os.path.join('./scripts')))
from predict import Predict

def record(duration=10, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording


def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


def app():
    st.write("""
        ### Real Time Amharic Speech Recognition
         - This feature allows the user to get translations while talking.
        """)
    predict = Predict()

    if st.button(f"Click to Record"):
        record_state = st.text("Recording...")
        duration = 14  # seconds
        fs = 48000
        myrecording = record(duration, fs)
        # myrecording = ""
        record_state.text(f"Transcribing...")
        path_myrecording = f"./data/wav/temp.mp3"
        save_record(path_myrecording, myrecording, fs)
        txt = predict.predict("./data/wav/temp.mp3")
        st.text(txt + "nothing")