from fastapi import FastAPI
import joblib
from preprocess import preprocess

app = FastAPI()
# model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    X = preprocess(data)
    # pred = model.predict(X)
    # return {"Survived": int(pred[0])}