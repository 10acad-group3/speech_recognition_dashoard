import streamlit as st

st.title('Speech Recognition Dashboard')

pages  = st.sidebar.radio('Page Navigation',['Data processing Steps','Try out the model'])

if pages == 'Data processing Steps':
    st.title('Audio Proprocessing Steps')
    col1,mid,col2 = st.columns([30,1,1])
    with col1:
       st.write('Step 1: Audio is Normalized')
       st.image('../images/pre1.PNG',width=800)

    col1,mid,col2 = st.columns([30,1,1])
    with col1:
        st.write('Step 2: Audio is Normalized')
        st.image('../images/pre1.PNG',width=800)

else:
    st.title('Try out the model')
