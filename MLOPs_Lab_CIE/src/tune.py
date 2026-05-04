import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json
import joblib

print("STARTING TASK 2")

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("resolution_hours", axis=1)
y = df["resolution_hours"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 7, 15],
    "min_samples_split": [2, 4]
}

model = RandomForestRegressor(random_state=42)

# IMPORTANT LINE (you were missing execution)
grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error"
)

# THIS LINE WAS MISSING ❗❗❗
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# Evaluate
preds = best_model.predict(X_test)
best_mae = mean_absolute_error(y_test, preds)
best_cv_mae = -grid.best_score_

# Save model
joblib.dump(best_model, "models/model.pkl")

# Save JSON
output = {
    "search_type": "grid",
    "n_folds": 3,
    "total_trials": len(grid.cv_results_["params"]),
    "best_params": grid.best_params_,
    "best_mae": best_mae,
    "best_cv_mae": best_cv_mae,
    "parent_run_name": "tuning-shieldops"
}

with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 Completed ✅")