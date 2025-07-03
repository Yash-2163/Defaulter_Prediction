
# Bank Defaulter Prediction System

A full‑stack machine learning application that predicts whether a bank customer will default on a personal loan. Includes prediction, data drift monitoring, and experiment tracking.

---

## Features

- Upload customer data in CSV format  
- Predict loan default using a trained BaggingClassifier model  
- Monitor data drift and generate data summaries using Evidently  
- Log metrics and experiments in MLflow  
- Interactive Streamlit frontend  
- Flask backend API for real‑time inference  
- Docker Compose for containerized deployment  

---

## Model Overview

- Algorithm: `BaggingClassifier` (scikit‑learn)  
- Preprocessing: missing‑value imputation + scaling  
- Hyperparameter tuning: GridSearchCV  
- Saved pipeline: `joblib`  

---

## Project Structure

```
Bank-Defaulter-Prediction-Project/
├── backend/                   # Flask API
│   ├── app.py
│   ├── model/
│   │   └── model.pkl
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                  # Streamlit prediction UI
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── DataDrift/                 # Drift detection & monitoring UI
│   ├── DataDriftCheck.py
│   ├── generate_drift_report.py
│   └── datadrift_report_app.py
├── model_training/            # Model training pipeline
│   └── train_model.py
├── data/                      # Datasets
│   ├── Personal_Loan.csv
│   └── newData.csv
├── docker-compose.yml         # Docker Compose config
└── README.md
```

---

## Getting Started

### Option 1: Run Locally (no Docker)

**Backend**  
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
API available at `http://localhost:5000`

**Frontend (Predictions)**  
```bash
cd frontend
python -m venv venv
# activate venv...
pip install -r requirements.txt
streamlit run streamlit_app.py
```
UI available at `http://localhost:8501`

**Drift Analysis UI**  
```bash
cd DataDrift
pip install -r requirements.txt
streamlit run datadrift_report_app.py
```

---

### Option 2: Run with Docker Compose

Make sure Docker is installed, then from project root:
```bash
docker-compose up --build
```
- Prediction UI → `http://localhost:8501`  
- Backend API → `http://localhost:5000`  

---

## CSV Input Format

Your CSV should **exclude** these columns:  
`ID`, `ZIP Code`, `Personal Loan` (target)  

Required columns example:  
```
Age,Experience,Income,Family,CCAvg,Education,Mortgage,Securities Account,CD Account,Online,CreditCard
```

---

## MLflow Tracking

Drift and summary metrics are logged to MLflow. To view:
```bash
mlflow ui
```
Open `http://localhost:5000` in your browser.

---

## License

This project is released under the MIT License.
