from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import pandas as pd

app = FastAPI(title="TrendX ML Predictor")

# Fake trained model for demo, you can train real one later
model = xgb.XGBClassifier()
# If you train and save: model.load_model("trendx_model.json")

class PriceData(BaseModel):
    prices: list[float]

@app.post("/predict")
def predict(data: PriceData):
    prices = np.array(data.prices)
    # feature example: mean change + std deviation
    features = np.array([[np.mean(np.diff(prices)), np.std(prices)]])
    
    # dummy prediction for now
    pred = np.random.choice([0,1])
    confidence = np.random.uniform(70, 95)
    predicted_price = prices[-1] + np.random.uniform(-0.5, 0.5)

    movement = "UP" if pred == 1 else "DOWN"

    return {
        "movement": movement,
        "confidence": round(confidence, 2),
        "predicted_price": round(predicted_price, 2)
    }