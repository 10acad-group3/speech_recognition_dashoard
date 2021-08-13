import streamlit as st


def app():
    # st.image('./img/rossman_store.jpeg')
    st.success('This demo is using Deep Learning Model to process and convert African language (Amharic) speech/voice to text format.')
    st.write(
        """
        #### To try it out,  
        - Please load the sample Amharic speech/audio files provided,
        - Upload your Amharic audio file, or
        - Use this demo to record your Speech and translate it
    """)
