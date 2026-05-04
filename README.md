# MLOPs Lab CIE

This repository contains the implementation of an end-to-end MLOps pipeline as part of the lab CIE. The workflow covers model training, hyperparameter tuning, retraining, evaluation, and containerization using Docker.

## Repository Structure

Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │   ├── training_data.csv
    │   └── new_data.csv
    ├── models/
    │   ├── model.pkl
    │   └── model_v2.pkl
    ├── results/
    │   ├── step1_s1.json
    │   ├── step2_s2.json
    │   ├── step3_s3.json
    │   └── step4_s4.json
    ├── src/
    │   ├── train.py
    │   ├── tune.py
    │   ├── retrain.py
    │   ├── step4.py
    │   └── predict_cli.py
    └── Dockerfile

## Workflow Overview

1. Model Training  
   The model is trained using the dataset in `training_data.csv`.

2. Hyperparameter Tuning  
   Different configurations are tested to improve model performance.

3. Model Retraining  
   The existing model is retrained using new data from `new_data.csv`, producing an updated model (`model_v2.pkl`).

4. Model Evaluation and Comparison  
   The old and new models are evaluated using metrics such as MAE and R² score. Results are stored in JSON format.

5. Docker Containerization  
   The application is containerized using a Dockerfile to enable reproducible execution.

## How to Run

### Train the Model
```bash
python src/train.py
