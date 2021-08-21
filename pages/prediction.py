import sounddevice as sd
import os
import sys
import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import librosa.display
sys.path.append(os.path.abspath(os.path.join('./scripts')))
from predict import Predict


def rerun():
    raise st.script_runner.RerunException(
        st.script_request_queue.RerunData(None))


def wav_plot(signal, title, x_label, y_label, sr=22000):
    fig = plt.figure(figsize=(25, 5))
    librosa.display.waveplot(signal, sr=sr)
    plt.title(title)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    st.pyplot(fig)


def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes


def app():
    predict = Predict()

    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    if st.session_state.audio_file is None:
        st.write(""" ### Upload Amharic audio file """)
        try:
            file = st.file_uploader("Pick a file")
            if file != None:
                if st.button('Click Here to Translate your Audio'):
                    audio_file = predict.get_audio(file)
                    st.session_state.audio_file = audio_file
                    rerun()
            else:
                st.error('Please, Upload the audio file')

        except Exception as e:
            print(f" Exception occurred in uploading audio file, {e}")
    else:
        st.write(""" ### Transcription """)
        clean_audio_file = predict.get_clean_audio(st.session_state.audio_file)
        txt = predict.predict(clean_audio_file)
        st.text(txt)
        st.session_state.translated = True

        if st.button('Visualize'):
            new_audio_file = read_audio(
                '/Users/ea/Desktop/speech_recognition_dashoard/data/AMHARIC/test/wav/01_d501021.wav')
            st.audio(new_audio_file)
            wav_plot(st.session_state.audio_file,
                     'Signal', 'Amplitude', 'Time (Sec)')
            wav_plot(clean_audio_file, 'Cleaned Signal',
                     'Amplitude', 'Time (Sec)')
            predict.get_spec(clean_audio_file)

        if st.button('Clear'):
            st.session_state.audio_file = None
            rerun()
