import os
from fastapi import FastAPI, HTTPException
import skops.io as skops
from app.preprocess import preprocess
from sklearn.base import BaseEstimator
from pydantic import BaseModel, Field, ValidationError

app = FastAPI()
MODEL_PATH = os.getenv("MODEL_PATH", "app/models/model.skops")
model: BaseEstimator = skops.load(MODEL_PATH)

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

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[0][1]  # вероятность выжить
        else:
            prob = None
        
        return {
            "Survived": int(pred[0]),
            "Probability": round(prob, 2) if prob is not None else None
        }
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")