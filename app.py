#app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import traceback
import warnings

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load all trained models and feature lists
models = {
    "random_forest": joblib.load("random_forest_model.pkl"),
    "svm": joblib.load("svm_model.pkl"),
    "neural_network": joblib.load("neural_network_model.pkl"),
    "logistic_regression": joblib.load("logistic_regression_model.pkl"),
    "knn": joblib.load("knn_model.pkl")
}

@app.route("/")
def home():
    return "Machine Learning Model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON from request
        data = request.get_json(force=True)
        model_name = data.get("model_name")
        input_data = data.get("input", {})

        print("\nIncoming Payload:", input_data)

        # Validate model selection
        if model_name not in models:
            return jsonify({"error": f"Invalid model name: {model_name}"}), 400

        # Load model and expected features
        model_bundle = models[model_name]
        model = model_bundle["model"]
        expected_features = model_bundle["features"]

        # Debug logs
        print("Expected features:", expected_features)
        print("Received fields:", list(input_data.keys()))

        # Check for missing input features
        missing = [col for col in expected_features if col not in input_data]
        if missing:
            return jsonify({"error": f"Missing required input fields: {missing}"}), 400

        # Arrange input in the correct order
        row = [input_data[col] for col in expected_features]

        # Convert to NumPy array and predict
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