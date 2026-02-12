# Titanic RandomForest ML API

A machine learning pet project in Python with RandomForest trained on the Titanic dataset.

The project includes:

- Model training (train.py)
- Simple inference API via FastAPI (app/main.py)
- Input data preprocessing (app/preprocess.py)
- Docker container for easy deployment

Project demonstrates the complete ML → MLOps cycle: data → model → API → Docker.

## Technologies used
- Python 3.11
- scikit-learn 
- Pandas
- FastAPI
- Joblib
- Docker

## How to Run

### Local Setup
```bash
git clone https://github.com/Olga-Shabatina/titanic-mlflow-sklearn-docker.git
cd titanic-mlflow-sklearn-docker
pip install -r requirements.txt
python train.py
uvicorn app.main:app --reload
```

### Run with Docker
```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```
### Test the API

#### Swagger UI

1. Open  browser and go to: http://127.0.0.1:8000/docs

2. Click on the `/predict` endpoint

3. Click "Try it out"

4. Paste the JSON example into the request body

5. Click "Execute"

### Curl

Run the following `curl` command in a terminal.

Linux:
```bash
curl -s -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"pclass":3,"sex":0,"age":25,"fare":8.05}' \
     -w "\n"
```

PowerShell (Windows):
```shell
Invoke-RestMethod `
   -Method POST `
   -Uri "http://127.0.0.1:8000/predict" `
   -ContentType "application/json" `
   -Body '{"pclass":1,"sex":0,"age":22,"fare":7.25}'
```

## API Request Example

**POST** `/predict`
```json
{
  "pclass": 1,
  "sex": "0",
  "age": 25,
  "sibsp": 0,
  "parch": 0,
  "fare": 28.05,
  "embarked": "S"
}
```
**Response:**
```json
{
  "Survived": 1,
  "Probability": 0.6
}
```
where `1` = survived, `0` = did not survive.