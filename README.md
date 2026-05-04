MLOPs Lab CIE
Overview

This repository contains the implementation of an end-to-end Machine Learning Operations (MLOps) pipeline developed as part of the lab Continuous Internal Evaluation (CIE). The project demonstrates the complete lifecycle of a machine learning model, including training, hyperparameter tuning, retraining with new data, evaluation, and deployment using Docker.

Project Structure

Internals_Basics/
└── MLOPs_Lab_CIE/
    ├── data/
    │    ├── training_data.csv
    │    └── new_data.csv
    ├── models/
    │    ├── model.pkl
    │    └── model_v2.pkl
    ├── results/
    │    ├── step1_s1.json
    │    ├── step2_s2.json
    │    ├── step3_s3.json
    │    └── step4_s4.json
    ├── src/
    │    ├── train.py
    │    ├── tune.py
    │    ├── retrain.py
    │    ├── step4.py
    │    └── predict_cli.py
    └── Dockerfile

Workflow
1. Model Training

The model is trained using the dataset provided in training_data.csv.

2. Hyperparameter Tuning

Different model configurations are tested to improve performance.

3. Model Retraining

The trained model is updated using new data from new_data.csv, and a new version (model_v2.pkl) is generated.

4. Model Evaluation

The performance of the old and new models is compared using metrics such as Mean Absolute Error (MAE) and R² score. Results are stored in JSON format.

5. Docker Containerization

The application is containerized using Docker to ensure portability and reproducibility.

Execution Steps

Run the following commands from the project root:

Train the model
python src/train.py

Tune hyperparameters
python src/tune.py

Retrain the model
python src/retrain.py

Evaluate models
python src/step4.py

Run predictions
python src/predict_cli.py --severity_level 3 --alerts_count 31 --analyst_experience 6 --is_automated 0

Docker Usage

Build Docker image
docker build -t mlops-lab .

Run container
docker run mlops-lab

Conclusion

This project demonstrates a complete MLOps workflow including model development, evaluation, versioning, and deployment using Docker.
