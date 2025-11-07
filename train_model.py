import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_and_save_model():
    # Define CSV path
    csv_path = os.path.join(os.getcwd(), "crops.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Load dataset
    df = pd.read_csv(csv_path)

    print("✅ Dataset loaded successfully. Shape:", df.shape)

    # Check required columns
    required_columns = ["temperature", "humidity", "moisture", "soil", "crop"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Encode soil_type
    soil_encoder = LabelEncoder()
    df["soil_type_encoded"] = soil_encoder.fit_transform(df["soil"])

    # Encode crop_type (target)
    crop_encoder = LabelEncoder()
    df["crop_type_encoded"] = crop_encoder.fit_transform(df["crop"])

    # Define features and target
    X = df[["temperature", "humidity", "moisture", "soil_type_encoded"]]
    y = df["crop_type_encoded"]

    # Train RandomForest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save model and encoders
    joblib.dump({
        "model": model,
        "soil_encoder": soil_encoder,
        "crop_encoder": crop_encoder
    }, "crop_model.pkl")

    print("✅ Model and encoders saved as crop_model.pkl")
    print("🌱 Unique Crops:", list(crop_encoder.classes_))
    print("🌍 Unique Soil Types:", list(soil_encoder.classes_))

if __name__ == "__main__":
    train_and_save_model()