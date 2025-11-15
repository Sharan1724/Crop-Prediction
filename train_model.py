import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_and_save_model():
    csv_path = os.path.join(os.getcwd(), "crops.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Dataset loaded →", df.shape)

    required_columns = ["temperature", "humidity", "moisture", "soil", "crop"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Encoders
    soil_encoder = LabelEncoder()
    df["soil_type_encoded"] = soil_encoder.fit_transform(df["soil"])

    crop_encoder = LabelEncoder()
    df["crop_type_encoded"] = crop_encoder.fit_transform(df["crop"])

    X = df[["temperature", "humidity", "moisture", "soil_type_encoded"]]
    y = df["crop_type_encoded"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save model (gzip-compressed)
    joblib.dump(
        {
            "model": model,
            "soil_encoder": soil_encoder,
            "crop_encoder": crop_encoder
        },
        "crop_model.pkl.gz",
        compress="gzip"
    )

    print("✅ Model saved → crop_model.pkl.gz")
    print("🌱 Crops:", list(crop_encoder.classes_))
    print("🌍 Soil Types:", list(soil_encoder.classes_))


if __name__ == "__main__":
    train_and_save_model()
