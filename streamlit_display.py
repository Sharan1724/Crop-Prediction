import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# ---------------------------
# MongoDB Connection
# ---------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["crop_prediction_db"]
collection = db["predictions"]

# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="🌾 Crop Predictions Dashboard", layout="wide")
st.title("🌾 Latest Crop Prediction Records")

# ---------------------------
# Load and Process Data
# ---------------------------
data = list(collection.find().sort("timestamp", -1).limit(30))

if data:
    records = []
    for d in data:
        # Handle both old and new field names
        temp = d.get("temperature") or d.get("inputs", {}).get("temperature")
        hum = d.get("humidity") or d.get("inputs", {}).get("humidity")
        moist = (
            d.get("moisture")
            or d.get("soil_moisture")
            or d.get("inputs", {}).get("moisture")
        )
        soil = (
            d.get("soil")
            or d.get("soil_type")
            or d.get("inputs", {}).get("soil")
        )
        crop = d.get("predicted_crop", "N/A")
        time = d.get("timestamp", "N/A")

        record = {
            "temperature": temp,
            "humidity": hum,
            "soil_moisture": moist,
            "soil_type": soil,
            "crop": crop,
            "Timestamp": time,
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Format timestamp for readability
    if "Timestamp" in df.columns:
        df["Timestamp"] = df["Timestamp"].apply(
            lambda t: datetime.fromisoformat(t).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(t, str) and t != "N/A"
            else t
        )

    st.success(f"✅ Connected to MongoDB | Showing latest {len(df)} records")

    # Display table neatly
    st.dataframe(df, use_container_width=True, hide_index=True)

    # CSV Download Option
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Latest 30 Records as CSV",
        data=csv,
        file_name="latest_crop_predictions.csv",
        mime="text/csv",
    )

else:
    st.info("No prediction records found. Run a crop prediction first to populate the database.")
