import os
import sys
import cv2
import pickle

import librosa
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio

import plotly.express as px
import matplotlib.pyplot as plt
from IPython.display import Image
import plotly.graph_objects as go
from keras.utils.vis_utils import plot_model
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from clean_audio import CleanAudio
from log_melgram_layer import LogMelgramLayer
from model import *


class Predict():
    def __init__(self):
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.clean_audio = CleanAudio()

    def get_model(self):
        fft_size = 256
        hop_size = 128
        n_mels = 128
        melspecModel = preprocessing_model(fft_size, hop_size, n_mels)
        resnet_, calc = resnet(n_mels, 224, 512, 4)
        model = build_model(melspecModel, 224, resnet_, calc)
        model.load_weights('./models/resnet_v3.h5')
        return model

    def get_tokenizer(self):
        with open('./models/char_tokenizer_amharic.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        return tokenizer

    def predict(self, audio_file):
        sr = 8000
        wav, rate = librosa.load(audio_file, sr=None)
        y = librosa.resample(wav, rate, sr)
        y = self.clean_audio.normalize_audio(y)
        y = self.clean_audio.split_audio(y, 30)
        y = y.reshape(1, -1)

        y_pred = self.model.predict(y)

        input_shape = tf.keras.backend.shape(y_pred)
        input_length = tf.ones(
            shape=input_shape[0]) * tf.keras.backend.cast(input_shape[1], 'float32')
        prediction = tf.keras.backend.ctc_decode(
            y_pred, input_length, greedy=False)[0][0]

        pred = K.eval(prediction).flatten().tolist()
        pred = list(filter(lambda a: a != -1, pred))

        return ''.join(self.tokenizer.tokens_to_string(pred))
