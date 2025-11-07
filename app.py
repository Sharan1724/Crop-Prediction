from flask import Flask, request, jsonify
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)

# -----------------------------
# ✅ Lazy model loading (saves memory)
# -----------------------------
model = None
soil_encoder = None
crop_encoder = None

def load_model():
    """Load model and encoders once (lazy load)"""
    global model, soil_encoder, crop_encoder
    if model is None:
        model_data = joblib.load("crop_model.pkl")
        model = model_data["model"]
        soil_encoder = model_data["soil_encoder"]
        crop_encoder = model_data["crop_encoder"]
        print("✅ Model and encoders loaded successfully.")


# -----------------------------
# ✅ MongoDB Atlas Connection
# -----------------------------
# Use environment variable for security
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://crop_db:<db_password>@crop-prediction.zfyxw9a.mongodb.net/?appName=Crop-Prediction")

client = MongoClient(MONGO_URI)
db = client["crop_prediction_db"]
collection = db["predictions"]

@app.route("/")
def home():
    return jsonify({"message": "🌾 Crop Prediction API is Running on Render!"})


# -----------------------------
# ✅ Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        load_model()  # Load model only once

        data = request.get_json(force=True)
        temperature = float(data.get("temperature"))
        humidity = float(data.get("humidity"))
        moisture = float(data.get("moisture"))
        soil_type = data.get("soil", "").strip().lower()

        # Encode soil type
        if soil_type not in soil_encoder.classes_:
            return jsonify({"error": f"Unknown soil type '{soil_type}'. Expected: {list(soil_encoder.classes_)}"}), 400

        soil_encoded = soil_encoder.transform([soil_type])[0]
        X = np.array([[temperature, humidity, moisture, soil_encoded]])

        # Predict probabilities
        probs = model.predict_proba(X)[0]
        top_indices = probs.argsort()[-5:][::-1]
        top_crops = crop_encoder.inverse_transform(top_indices)
        top_probs = [round(float(p * 100), 2) for p in probs[top_indices]]  # Suitability %

        # MongoDB record
        record = {
            "temperature": temperature,
            "humidity": humidity,
            "soil_moisture": moisture,
            "soil_type": soil_type,
            "predicted_crop": top_crops[0],
            "timestamp": datetime.utcnow().isoformat()
        }
        collection.insert_one(record)

        return jsonify({
            "status": "success",
            "top_crop": top_crops[0],
            "top_5_crops": [
                {"crop": crop, "suitability": f"{prob} %"}
                for crop, prob in zip(top_crops, top_probs)
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# ✅ Entry Point for Render
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)


