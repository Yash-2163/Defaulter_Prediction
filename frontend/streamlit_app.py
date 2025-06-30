# frontend/streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import io

st.set_page_config(page_title="Bank Defaulter Prediction", layout="centered")

st.title("🏦 Bank Defaulter Prediction System")
st.write("Upload a CSV file with customer data to predict defaulters.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        st.info("File uploaded successfully. Sending to backend for prediction...")

        # Properly package the file for requests
        files = {
            'file': (uploaded_file.name, uploaded_file, 'text/csv')
        }

        # Send POST request to Flask backend
        response = requests.post("http://backend:5000/predict", files=files)

        if response.status_code == 200:
            predictions = pd.read_json(io.StringIO(response.text))
            st.success("✅ Predictions received! Displaying results:")
            st.dataframe(predictions, use_container_width=True)
        else:
            st.error(f"❌ Backend error: {response.json().get('error')}")
    except Exception as e:
        st.error(f"🚨 Something went wrong: {str(e)}")

