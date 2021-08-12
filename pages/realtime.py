import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle


@st.cache
def loadModel():
    df = pd.read_csv("./pages/train.csv")  # for performance
    return df


def app():
    st.title('Sales Forcasting')
