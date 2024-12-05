import os
import streamlit as st

# Set your Google API key here
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
