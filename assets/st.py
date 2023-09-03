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


if st.button("Train Model"):

    response = requests.post('http://localhost:5000/train', json=Param)

    if response.status_code == 200:
        st.success("Training completed successfully!")

    else:
        st.error("Training failed. Check the Flask API.")

if st.button("Fetch Metrics"):

    response = requests.get('http://localhost:5000/metrics')

    if response.status_code == 200:
        metrics_data = response.json()
        st.write(metrics_data)  

        
        
    
        epochs = metrics_data["epochs"]
        accuracy = metrics_data["accuracy"]
        precision = metrics_data["precision"]
        recall = metrics_data["recall"]
        f1_score = metrics_data["f1_score"]
        years = metrics_data["years"]
        breach_count = metrics_data["breach_count"]
        correlation = metrics_data["correlation"]

       
        st.subheader("Metrics Over Epochs")
        st.line_chart(pd.DataFrame({
            'Epochs': epochs,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }))

      
        st.subheader("Breach Count and Correlation Over Years")
        st.line_chart(pd.DataFrame({
            'Year': years,
            'Breach Count': breach_count,
            'Correlation': correlation
        }))

    else:
        st.error("Failed to fetch metrics from the Flask API.")
