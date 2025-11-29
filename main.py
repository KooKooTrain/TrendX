import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from predictor import (
    InvalidTickerError,
    accuracyScore,
    get_df_and_model,
    predict_next_day,
)


async def stockInfo(request: Request):
    ticker = request.path_params["ticker"]

    try:
        (df, ticker), model, (x_test, y_test) = get_df_and_model(ticker)
    except InvalidTickerError:
        return JSONResponse(
            {
                "message": "Invalid ticker provided, please check the spelling and try again later."
            },
            status_code=400,
        )
    except Exception as e:
        return JSONResponse({"message": e})

    label = predict_next_day(model, x_test.iloc[[-1]])
    prices = df["Close"]

    company = ticker.info.get("longName") or ticker.info.get("displayName")
    symbol = ticker.ticker
    currency = ticker.info.get("financialCurrency") or "USD"
    exchange = ticker.info.get("exchange") or ""
    summary = ticker.info.get("longBusinessSummary") or ""

    return JSONResponse(
        {
            "company": company,
            "summary": summary,
            "symbol": symbol,
            "currency": currency,
            "exchange": exchange,
            "currentPrice": prices.iloc[-1].item(),
            "dailyChangePercent": (prices.pct_change() * 100).iloc[-1].item(),
            "sentiment": "UP" if label == 1 else "DOWN",
            "accuracyScore": accuracyScore(model, x_test, y_test),
            "featureSet": ["Momentum", "Volume", "MACD"],
            "trainingWindow": "1 Year",
        }
    )


async def stockPrices(request: Request):
    ticker = request.path_params["ticker"]

    try:
        (df, _), _, _ = get_df_and_model(ticker)
    except InvalidTickerError:
        return JSONResponse(
            {
                "message": "Invalid ticker provided, please check the spelling and try again later."
            },
            status_code=400,
        )
    except Exception as e:
        return JSONResponse({"message": e})

    last_30_prices = df["Close"].squeeze().tail(30).tolist()
    return JSONResponse({"prices": last_30_prices})


stocks_routes = Mount(
    "/api/stocks/{ticker:str}",
    routes=[
        Route("/info", stockInfo),
        Route("/prices", stockPrices),
    ],
)


async def delModel(request: Request):
    ticker = request.path_params["ticker"]
    get_df_and_model(ticker, True)

    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    app = Starlette(
        debug=True,
        routes=[
            Route("/api/model/{ticker}", delModel, methods=["DELETE"]),
            stocks_routes,
            Mount("/", app=StaticFiles(directory="static", html=True), name="static"),
        ],
    )

    uvicorn.run(app)
