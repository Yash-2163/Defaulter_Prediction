# frontend/streamlit_app.py

import streamlit as st
import pandas as pd
import requests
import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.animation import FuncAnimation
import tempfile
import numpy as np
from utils.clustering_utils import run_kmeans_animation
# from utils.clustering_utils import get_final_kmeans_clusters



st.set_page_config(page_title="Bank Defaulter Prediction App", layout="centered")

st.title("🏦 Bank Defaulter Prediction & Clustering System")

tab1, tab2 = st.tabs(["🔍 Predict Defaulters", "📊 Clustering Explorer"])

# --- TAB 1: Defaulter Prediction ---
with tab1:
    st.header("Upload CSV to Predict Defaulters")
    uploaded_file = st.file_uploader("Upload customer data CSV", type=["csv"], key="predict_csv")

    if uploaded_file:
        try:
            st.info("📤 Sending file to backend for prediction...")

            files = {
                'file': (uploaded_file.name, uploaded_file, 'text/csv')
            }

            response = requests.post("http://backend:5000/predict", files=files)

            if response.status_code == 200:
                predictions = pd.read_json(io.StringIO(response.text))
                st.success("✅ Predictions received!")
                st.dataframe(predictions, use_container_width=True)
            else:
                st.error(f"❌ Backend error: {response.json().get('error')}")
        except Exception as e:
            st.error(f"🚨 Something went wrong: {str(e)}")

# --- TAB 2: Clustering ---
with tab2:
    st.header("Upload Data for Clustering")

    clustering_file = st.file_uploader("Upload CSV with numerical features", type=["csv"], key="cluster_csv")

    if clustering_file:
        try:
            # ✅ Make sure df is defined immediately after reading the file
            df = pd.read_csv(clustering_file)
            st.subheader("📄 Uploaded Data")
            st.dataframe(df.head())

            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

            if len(numeric_cols) < 2:
                st.warning("❗Please upload a dataset with at least 2 numerical features for clustering.")
            else:
                feature1 = st.selectbox("Select X-axis feature", numeric_cols, index=0)
                feature2 = st.selectbox("Select Y-axis feature", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

                k = st.slider("Select number of clusters (K)", min_value=2, max_value=10, value=3)

                if st.button("Run Clustering"):
                    gif_path = run_kmeans_animation(df, feature1, feature2, k)
                    st.image(gif_path, caption="📈 Clustering Animation")
                    df_clustered = get_final_kmeans_clusters(df, feature1, feature2, k)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

