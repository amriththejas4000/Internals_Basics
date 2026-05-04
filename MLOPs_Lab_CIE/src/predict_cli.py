import argparse
import pandas as pd
import joblib

model = joblib.load("models/model.pkl")

parser = argparse.ArgumentParser()

parser.add_argument("--severity_level", type=int, required=True)
parser.add_argument("--alerts_count", type=int, required=True)
parser.add_argument("--analyst_experience", type=int, required=True)
parser.add_argument("--is_automated", type=int, required=True)

args = parser.parse_args()

data = pd.DataFrame([{
    "severity_level": args.severity_level,
    "alerts_count": args.alerts_count,
    "analyst_experience": args.analyst_experience,
    "is_automated": args.is_automated
}])

prediction = model.predict(data)[0]

print(prediction)