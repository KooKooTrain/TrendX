import pickle
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.endpoints import HTTPEndpoint
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from xgboost import XGBClassifier

from predictor import accuracyScore, df, predict_next_day

model = XGBClassifier()
model.load_model("./model.ubj")

with Path("./scaler.pkl").open("rb") as f:
    scaler, x_scaled, y_test = pickle.load(f)

if df is None:
    exit(1)


prices = df["Close"]
last_30_prices = df["Close"].squeeze().tail(30).tolist()


class StockInfo(HTTPEndpoint):
    async def get(self, request):
        label = predict_next_day(model, scaler, x_scaled[-1:])
        return JSONResponse(
            {
                "company": "Tesla Inc",
                "symbol": "TSLA",
                "currentPrice": prices.iloc[-1].item(),
                "dailyChangePercent": (prices.pct_change() * 100).iloc[-1].item(),
                "sentiment": "UP" if label == 1 else "DOWN",
                "accuracyScore": accuracyScore(model, x_scaled, y_test),
            }
        )


class StockPrices(HTTPEndpoint):
    async def get(self, request):
        return JSONResponse({"prices": last_30_prices})


if __name__ == "__main__":
    app = Starlette(
        debug=True,
        routes=[
            Route("/api/stock/info", StockInfo),
            Route("/api/stock/prices", StockPrices),
            Mount("/", app=StaticFiles(directory="static", html=True), name="static"),
        ],
    )

    uvicorn.run(app)
