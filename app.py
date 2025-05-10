# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import warnings

app = Flask(__name__)
CORS(app)

# Load all models (each includes model + features)
models = {
    "random_forest": joblib.load("random_forest_model.pkl"),
    "svm": joblib.load("svm_model.pkl"),
    "neural_network": joblib.load("neural_network_model.pkl"),
    "logistic_regression": joblib.load("logistic_regression_model.pkl"),
    "knn": joblib.load("knn_model.pkl")
}

@app.route("/")
def home():
    return "ML Model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        model_name = data.get("model_name")
        input_data = data.get("input", {})

        print("\nIncoming Payload:")
        print(input_data)

        if model_name not in models:
            return jsonify({"error": f"Invalid model name: {model_name}"}), 400

        # Load model and feature list
        bundle = models[model_name]
        model = bundle["model"]
        expected_features = bundle["features"]

        # Ensure all required fields are present
        missing = [col for col in expected_features if col not in input_data]
        if missing:
            return jsonify({"error": f"Missing required input fields: {missing}"}), 400

        # Construct input row in order
        row = [input_data[col] for col in expected_features]

        # Predict
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = int(model.predict([row])[0])

        return jsonify({
            "model": model_name,
            "prediction": prediction
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)