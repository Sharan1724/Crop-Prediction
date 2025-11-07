from flask import Flask, request, jsonify
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)

# -----------------------------
# Load trained model and encoders
# -----------------------------
model_data = joblib.load("crop_model.pkl.gz")
model = model_data["model"]
soil_encoder = model_data["soil_encoder"]
crop_encoder = model_data["crop_encoder"]

# -----------------------------
# MongoDB Atlas setup
# -----------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://crop_db:<db_password>@crop-prediction.zfyxw9a.mongodb.net/?appName=Crop-Prediction")
client = MongoClient(MONGO_URI)
db = client["crop_prediction_db"]
collection = db["predictions"]

@app.route('/')
def home():
    return "🌾 Crop Prediction API is Running on Render!"

@app.route('/predict', methods=['POST'])
def predict_crop():
    try:
        data = request.get_json()

        # Extract input values
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        moisture = float(data["moisture"])
        soil_type = data["soil"].strip().lower()

        # Encode soil type
        if soil_type not in soil_encoder.classes_:
            return jsonify({
                "error": f"Unknown soil type '{soil_type}'. Expected: {list(soil_encoder.classes_)}"
            }), 400

        soil_encoded = soil_encoder.transform([soil_type])[0]
        X = np.array([[temperature, humidity, moisture, soil_encoded]])

        # Predict probabilities
        probs = model.predict_proba(X)[0]
        top_indices = probs.argsort()[-5:][::-1]
        top_crops = crop_encoder.inverse_transform(top_indices)
        top_crop = top_crops[0]

        # Save to MongoDB Atlas
        record = {
            "temperature": temperature,
            "humidity": humidity,
            "soil_moisture": moisture,
            "soil_type": soil_type,
            "predicted_crop": top_crop,
            "timestamp": datetime.now().isoformat()
        }
        collection.insert_one(record)

        return jsonify({
            "status": "success",
            "top_5_crops": top_crops.tolist(),
            "top_crop": top_crop
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


