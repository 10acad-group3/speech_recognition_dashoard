import os
import sys
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import pickle
import wavio
import numpy as np
import streamlit as st
import sounddevice as sd
sys.path.append(os.path.abspath(os.path.join('./scripts')))
from predict import Predict

duration = 5  # seconds
fs = 48000


def rerun():
    raise st.script_runner.RerunException(
        st.script_request_queue.RerunData(None))


def record():
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    path_myrecording = "./data/wav/temp.mp3"
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return myrecording


def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

@st.cache
def getPredictor():
    predict = Predict()
    return predict


def app():
    st.write(""" ### Real Time Amharic Speech Recognition """)
    predict = getPredictor()

    if 'recording_state' not in st.session_state:
        st.session_state.recording_state = False

    if(not st.session_state.recording_state):
        btn = st.button("Start Recording")
        if btn:
            st.session_state.recording_state = True
            rerun()

    elif(st.session_state.recording_state):
        st.write("Recording audio...")

        myrecording = record()
        sd.play(myrecording, fs)  # st
        st.write(""" ### Transcription """)
        audio_file = read_audio('./data/wav/temp.mp3')
        st.audio(audio_file)
        audio_file = predict.get_audio('./data/wav/temp.mp3')
        clean_audio_file = predict.get_clean_audio(audio_file)
        txt = predict.predict(clean_audio_file)
        if(txt == ""):
            st.text("Please speak louder")
        else:
            st.text(txt)
        btn = st.button("Clear Output")    
        if btn:
            st.session_state.recording_state = False
            rerun()

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
