from fastapi import FastAPI, HTTPException
import joblib
from app.preprocess import preprocess
from sklearn.base import BaseEstimator
from pydantic import BaseModel, Field, ValidationError

app = FastAPI()
model: BaseEstimator = joblib.load("app/model.pkl")

class Passenger(BaseModel):
    pclass: int = Field(..., ge = 1, le = 3, description = "class: 1, 2, 3")
    sex: int = Field(..., ge = 0, le = 1, escription = "0 = male, 1 = female")
    age: float = Field(..., ge = 0, le = 130)
    fare: float = Field(..., ge = 0)

@app.post("/predict")
def predict(passenger: Passenger):
    try:
        X = preprocess(passenger.model_dump())
        pred = model.predict(X)
        return {"Survived": int(pred[0])}
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")