# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # Allow requests from Streamlit frontend

# Load model pipeline
model_path = os.path.join("model", "model.pkl")
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        input_df = pd.read_csv(file)
        predictions = model.predict(input_df)
        input_df['Predicted Defaulter'] = predictions
        return input_df.to_json(orient='records')  # send list of dicts
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

