# Titanic RandomForest ML API

A machine learning pet project that predicts Titanic passenger survival using RandomForest. Model inference is served via FastAPI, containerized with Docker.

## Features
- Trained RandomForest classifier on Titanic dataset
- REST API endpoint for predictions (`/predict`)
- Containerized with Docker for easy deployment

## How to Run

### Local Setup
```bash
git clone https://github.com/Olga-Shabatina/titanic-mlflow-docker.git
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

### Run with Docker
```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

## API Request Example

**POST** `/predict`
```json
{
  "pclass": 3,
  "sex": "male",
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 8.05,
  "embarked": "S"
}
```
**Response:**
```json
{
  "survived": 0,
  "probability": 0.21
}
```
