import pandas as pd
import joblib

print("START RETRAIN")

# Load model
model = joblib.load("models/model.pkl")

# Load new data
df = pd.read_csv("data/new_data.csv")

print("DATA LOADED:", df.shape)

X_new = df.drop("resolution_hours", axis=1)
y_new = df["resolution_hours"]

print("TRAINING...")

# Retrain
model.fit(X_new, y_new)

# Save
joblib.dump(model, "models/model_v2.pkl")

print("Retraining Done ✅")