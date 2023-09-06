from ast import Param
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Intrusion Detection System",
    page_icon="🚀",
    layout="centered"
)

st.title("Intrusion Detection System")

try:
    if st.button("Train Model"):
        response = requests.post('http://localhost:5000/train', json=Param)

        if response.status_code == 200:
            st.success("Training completed successfully!")
        else:
            st.error("Training failed. Check the Flask API.")
except requests.exceptions.RequestException as e:
    st.error(f"Failed to connect to the Flask API: {e}")

try:
    if st.button("Fetch Metrics"):
        response = requests.get('http://localhost:5000/metrics')

        if response.status_code == 200:
            metrics_data = response.json()
            st.write(metrics_data) 
        else:
            st.error("Failed to fetch metrics from the Flask API.")
except requests.exceptions.RequestException as e:
    st.error(f"Failed to connect to the Flask API: {e}")
