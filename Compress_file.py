import joblib

# Full path to your original model
input_path = r"D:\Road_Risk_Predictor_Using_Machine_Learning\ML_server2\random_forest_model.pkl"

# Load the existing model
model = joblib.load(input_path)

# Re-save the model in the same path with compression
joblib.dump(model, input_path, compress=3)

print("Model compressed and overwritten at:", input_path)