FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install pandas scikit-learn joblib

CMD ["python3", "src/predict_cli.py"]