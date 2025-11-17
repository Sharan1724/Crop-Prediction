from flask import Flask, request, jsonify
import joblib
import numpy as np
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
        model_data = joblib.load("crop_model.pkl.gz")
        model = model_data["model"]
        soil_encoder = model_data["soil_encoder"]
        crop_encoder = model_data["crop_encoder"]
        print("✅ Model loaded successfully.")


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

        data = request.get_json(force=True)

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        moisture = float(data["moisture"])
        soil_type = data["soil"].strip().lower()

        # Validate soil type
        if soil_type not in soil_encoder.classes_:
            return jsonify({
                "error": f"Unknown soil type '{soil_type}'",
                "valid_soil_types": list(soil_encoder.classes_)
            }), 400

        # Encode soil type
        soil_encoded = soil_encoder.transform([soil_type])[0]

        # Feature vector
        X = np.array([[temperature, humidity, moisture, soil_encoded]])

        # Predict probabilities
        probs = model.predict_proba(X)[0]

        # Sort highest → lowest
        sorted_idx = np.argsort(-probs)

        # Always send exactly 5 crops
        top_indices = sorted_idx[:5]

        # Decode crops
        top_crops = crop_encoder.inverse_transform(top_indices)

        # Return result to MIT App Inventor
        return jsonify({
            "top_crops": list(top_crops)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Railway Entry Point
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
