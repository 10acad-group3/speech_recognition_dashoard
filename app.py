import os
import sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join('./pages')))
import home
import about
import realtime
import prediction
from tokenizer import TokenizerWrap


PAGES = {
    "Home": home,
    "Prediction": prediction,
    "Real Time Prediction": realtime,
    "About": about
}

st.set_page_config(
    page_title="Dashboard | Amharic Speech Recognition ", layout="wide")
st.markdown("<h1 style='color:#0b4eab;font-size:36px;border-radius:10px;'>Dashboard | Amharic Speech Recognition </h1>", unsafe_allow_html=True)

selection = st.sidebar.radio("Go to page", list(PAGES.keys()))
page = PAGES[selection]
page.app()
