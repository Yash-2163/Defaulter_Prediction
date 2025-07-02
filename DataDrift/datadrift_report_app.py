# DataDrift/datadrift_report_app.py

import streamlit as st
from DataDriftCheck import generate_all_reports

st.set_page_config(page_title="Data Drift & Summary", layout="wide")

def display_html(path):
    with open(path, "r", encoding="utf-8") as f:
        st.components.v1.html(f.read(), height=900, scrolling=True)

def main():
    st.title("📊 Data Drift & Data Summary Dashboard")
    st.info("Running reports...")

    reports = generate_all_reports()
    st.success("✅ Reports ready!")

    choice = st.sidebar.radio("Select report:", list(reports.keys()))
    st.header(f"{choice} Report")
    display_html(reports[choice])

if __name__ == "__main__":
    main()
