import os
import sys
import wavio
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
        st.write("Result")
        txt = predict.predict("./data/wav/temp.mp3")
        st.text(txt + "nothing")

        btn = st.button("Clear Output")
        if btn:
            st.session_state.recording_state = False
            rerun()
