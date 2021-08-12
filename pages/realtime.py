import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import wavio
import sounddevice as sd


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
    st.title('Sales Forcasting')

    filename = st.text_input("Choose a filename: ")

    if st.button(f"Click to Record"):
        if filename == "":
            st.warning("Choose a filename.")
        else:
            record_state = st.text("Recording...")
            duration = 5  # seconds
            fs = 48000
            myrecording = record(duration, fs)
            record_state.text(f"Saving sample as {filename}.wav")

            path_myrecording = f"./data/wav/{filename}.wav"

            save_record(path_myrecording, myrecording, fs)
