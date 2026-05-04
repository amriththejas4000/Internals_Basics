# MLOps Lab CIE

This project implements a complete end-to-end MLOps pipeline for predicting **resolution time (hours)** using machine learning models.

---

## Project Overview

The pipeline includes:

* Model training with experiment tracking
* Hyperparameter tuning
* Model retraining with new data
* Model comparison and evaluation
* CLI-based prediction
* Docker containerization

---

## Project Structure

```
MLOPs_Lab_CIE/
│
├── data/               # Training and new datasets
├── mlruns/            # MLflow experiment tracking
├── models/            # Saved models (model.pkl, model_v2.pkl)
├── results/           # Output JSON files for each step
├── src/               # Source code
│   ├── train.py
│   ├── tune.py
│   ├── retrain.py
│   ├── step4.py
│   ├── predict_cli.py
│
├── Dockerfile         # Docker configuration
└── README.md
```

---

## Steps Implemented

### Step 1: Model Training

* Trained multiple models (Lasso, RandomForest)
* Logged experiments using MLflow
* Evaluated using:

  * MAE
  * RMSE
  * R²
  * MAPE
* Selected best model based on MAE

---

### step 2: Hyperparameter Tuning

* Used GridSearchCV
* 3-fold cross-validation
* Optimized model parameters
* Saved best model and results

---

### Step 3: Model Retraining

* Used new incoming data
* Retrained model
* Saved updated model (`model_v2.pkl`)

---

### Step 4: Model Comparison

* Compared old vs retrained model
* Metrics used:

  * MAE
  * R²
* Determined whether new model improves performance

---

## Docker Usage

### Build Docker Image

```
docker build -t mlops-predictor .
```

### Run Prediction

```
docker run mlops-predictor \
--severity_level 3 \
--alerts_count 31 \
--analyst_experience 6 \
--is_automated 0
```

---

## Run Locally

### Train Model

```
python3 src/train.py
```

### Tune Model

```
python3 src/tune.py
```

### Retrain Model

```
python3 src/retrain.py
```

### Compare Models

```
python3 src/step4.py
```

### Predict via CLI

```
python3 src/predict_cli.py \
--severity_level 3 \
--alerts_count 31 \
--analyst_experience 6 \
--is_automated 0
```

---

## Output Files

* `results/step1_s1.json` → Training results
* `results/step2_s2.json` → Tuning results
* `results/step3_s3.json` → Retraining info
* `results/step4_s4.json` → Model comparison

---

## Key Concepts Used

* Machine Learning (Regression)
* Hyperparameter Tuning
* Cross Validation
* MLflow Experiment Tracking
* Docker Containerization
* CLI-based Model Inference

---

## Conclusion

This project demonstrates a complete MLOps workflow from model development to deployment, ensuring reproducibility, scalability, and performance tracking.


