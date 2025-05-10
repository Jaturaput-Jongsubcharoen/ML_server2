# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import warnings
import os

app = Flask(__name__)
CORS(app)

MODEL_FILES = {
    "random_forest": "random_forest_model.pkl",
    "svm": "svm_model.pkl",
    "neural_network": "neural_network_model.pkl",
    "logistic_regression": "logistic_regression_model.pkl",
    "knn": "knn_model.pkl"
}

def load_model_bundle(model_name):
    """Load the model and its expected feature list."""
    path = MODEL_FILES.get(model_name)
    if path and os.path.exists(path):
        bundle = joblib.load(path)
        if isinstance(bundle, dict) and "model" in bundle and "features" in bundle:
            return bundle["model"], bundle["features"]
        else:
            raise ValueError(f"Invalid model format in {path}")
    else:
        raise FileNotFoundError(f"Model file not found: {path}")

@app.route("/")
def home():
    return "ML Model API is running."

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return '', 204

    try:
        data = request.get_json(force=True)
        model_name = data.get("model_name")
        input_data = data.get("input", {})

        if model_name not in MODEL_FILES:
            return jsonify({"error": f"Invalid model name: {model_name}"}), 400

        model, expected_features = load_model_bundle(model_name)

        print("Expected features:", expected_features)
        print("Received:", list(input_data.keys()))

        missing = [f for f in expected_features if f not in input_data]
        if missing:
            return jsonify({"error": f"Missing required input fields: {missing}"}), 400

        row = [input_data[f] for f in expected_features]

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
