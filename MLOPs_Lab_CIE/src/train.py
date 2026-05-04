import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("resolution_hours", axis=1)
y = df["resolution_hours"]

# Split (VERY IMPORTANT as per question)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("shieldops-resolution-hours")

results = []

def evaluate(model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

        mlflow.set_tag("team", "ml_engineering")

        mlflow.sklearn.log_model(model, name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

# Train models
evaluate(Lasso(), "Lasso")
evaluate(RandomForestRegressor(random_state=42), "RandomForest")

# Select best model
best = min(results, key=lambda x: x["mae"])

output = {
    "experiment_name": "shieldops-resolution-hours",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 Completed ✅")