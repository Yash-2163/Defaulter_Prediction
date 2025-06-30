# 🏦 Bank Defaulter Prediction System

A full-stack ML web application to predict bank customer defaults based on personal and financial attributes.

This project includes:
- 🔍 ML model: Trained BaggingClassifier with preprocessing pipelines.
- ⚙️ Flask backend: Serves the model for real-time inference.
- 💻 Streamlit frontend: Upload CSVs and view predictions interactively.
- 🐳 Dockerized setup: Easily deployable with Docker Compose.

---

## 🚀 Features

- Upload a CSV with customer data.
- Get predictions on who is likely to default.
- User-friendly interactive frontend.
- API-based Flask backend using a saved ML pipeline.
- Fully containerized using Docker.

---

## 🧠 Model

The model is a BaggingClassifier trained using scikit-learn pipelines. It includes:
- Preprocessing (imputation + scaling)
- Hyperparameter tuning using GridSearchCV
- Model serialization using joblib

---

## 📁 Project Structure

Bank-Defaulter-Prediction-Project/
├── backend/
│   ├── app.py
│   ├── model/
│   │   └── model.pkl
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   └── Personal_Loan.csv
├── model_training/
│   └── train_model.py
├── docker-compose.yml
└── README.md

---

## ⚙️ Getting Started

### 🔧 Option 1: Local Setup (without Docker)

Backend:
cd backend
python -m venv venv
venv\Scripts\activate  (or source venv/bin/activate)
pip install -r requirements.txt
python app.py  (Runs on http://localhost:5000)

Frontend:
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py  (Opens at http://localhost:8501)

---

### 🐳 Option 2: Run with Docker

Ensure Docker is installed, then:

docker-compose up --build

- Visit frontend: http://localhost:8501  
- Backend runs at: http://localhost:5000 (used internally)

---

## 📤 CSV Input Format

Your input CSV must not include ID, ZIP Code, or Defaulter.

Example columns:
Age,Experience,Income,Family,CCAvg,Education,Mortgage,Securities Account,CD Account,Online,CreditCard

---

## 📦 Dependencies

Backend:
- Flask
- flask-cors
- pandas, numpy
- scikit-learn
- joblib

Frontend:
- streamlit
- pandas
- requests

---

## 🙋‍♂️ Author

Yash Rajput

---

## 📄 License
