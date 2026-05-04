import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import json

print("START STEP 4")

# Load new data
df = pd.read_csv("data/new_data.csv")

X = df.drop("resolution_hours", axis=1)
y = df["resolution_hours"]

# Load models
old_model = joblib.load("models/model.pkl")
new_model = joblib.load("models/model_v2.pkl")

# Predictions
old_preds = old_model.predict(X)
new_preds = new_model.predict(X)

# Metrics
old_mae = mean_absolute_error(y, old_preds)
new_mae = mean_absolute_error(y, new_preds)

old_r2 = r2_score(y, old_preds)
new_r2 = r2_score(y, new_preds)

# Compare
improved = new_mae < old_mae

# Save result
output = {
    "old_model": {"mae": old_mae, "r2": old_r2},
    "new_model": {"mae": new_mae, "r2": new_r2},
    "improved": improved
}

with open("results/step4_s4.json", "w") as f:
    json.dump(output, f, indent=4)

print("Step 4 Completed ✅")