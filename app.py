from flask import Flask, request, jsonify
import joblib
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)

# -----------------------------
# Lazy model loading
# -----------------------------
model = None
soil_encoder = None
crop_encoder = None

def load_model():
    global model, soil_encoder, crop_encoder
    if model is None:
        # load compressed gzip model
        model_data = joblib.load("crop_model.pkl.gz")
        model = model_data["model"]
        soil_encoder = model_data["soil_encoder"]
        crop_encoder = model_data["crop_encoder"]
        print("Model loaded successfully.")

# -----------------------------
# MongoDB
# -----------------------------
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://crop_db:<db_password>@crop-prediction.zfyxw9a.mongodb.net/?retryWrites=true&w=majority"
)

client = MongoClient(MONGO_URI)
db = client["crop_prediction_db"]
collection = db["predictions"]


@app.route("/")
def home():
    return jsonify({"message": "🌾 Crop Prediction API is Running!"})


# -----------------------------
# Prediction API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_crop():
    try:
        load_model()

        # Force JSON decoding
        data = request.get_json(silent=False, force=True)

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        moisture = float(data["moisture"])
        soil_type = data["soil"].strip().lower()

        # Validate soil
        if soil_type not in soil_encoder.classes_:
            return jsonify({
                "error": f"Unknown soil type '{soil_type}'.",
                "expected": list(soil_encoder.classes_)
            }), 400

        soil_encoded = soil_encoder.transform([soil_type])[0]

        X = np.array([[temperature, humidity, moisture, soil_encoded]])

        # Predict top crops
        probs = model.predict_proba(X)[0]
        top_indices = probs.argsort()[-5:][::-1]
        top_crops = crop_encoder.inverse_transform(top_indices)

        # Save to MongoDB
        collection.insert_one({
            "temperature": temperature,
            "humidity": humidity,
            "soil_moisture": moisture,
            "soil_type": soil_type,
            "predicted_crop": top_crops[0],
            "timestamp": datetime.utcnow().isoformat()
        })

        return jsonify({"top_crops": list(top_crops)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)

