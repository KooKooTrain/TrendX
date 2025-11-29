import pickle  # noqa: S403
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def create_features(df: pd.DataFrame, target_col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    windows = [3, 5, 10, 20, 50]
    for w in windows:
        # Standard rolling features
        df[f"roll_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"roll_std_{w}"] = df[target_col].rolling(w).std()
        df[f"roll_min_{w}"] = df[target_col].rolling(w).min()
        df[f"roll_max_{w}"] = df[target_col].rolling(w).max()
        df[f"pct_change_{w}"] = df[target_col].pct_change(w)
        df[f"momentum_{w}"] = df[target_col] - df[target_col].shift(w)
        df[f"vol_mean_{w}"] = df["Volume"].rolling(w).mean()
        df[f"vol_std_{w}"] = df["Volume"].rolling(w).std()

    # Price-based features
    df["OC_diff"] = df["Close"] - df["Open"]
    df["HL_range"] = df["High"] - df["Low"]

    # Lag features
    for lag in range(1, 6):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    # MACD
    df["EMA12"] = df[target_col].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df[target_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df.dropna()


def create_labels(
    df: pd.DataFrame, target_col: str = "Close", threshold: float = 0.001
) -> pd.DataFrame:
    df = df.copy()
    # Create classification labels for up or down
    df["target"] = (
        # Use a threshold to reduce noise
        (df[target_col].shift(-1) - df[target_col]) / df[target_col] > threshold
    ).astype(int)
    return df.dropna()


def scale_data(x_train: list, x_test: list):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = df.drop(columns="target")
    y = df["target"]
    return cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]",
        train_test_split(x, y, test_size=0.2, shuffle=False),
    )


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    # First scale data and then make a base model on the scaled data
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "xgb",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )

    # Hypertune the model's parameters for maximum accuracy
    param_distributions = {
        "xgb__n_estimators": [300, 500, 800],
        "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "xgb__max_depth": [3, 4, 5, 6],
        "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
        "xgb__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "xgb__gamma": [0, 0.1, 0.3, 0.5],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__reg_alpha": [0, 0.1, 1],
        "xgb__reg_lambda": [1, 1.5, 2],
    }

    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=25,
        scoring="accuracy",
        cv=tscv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    search.fit(x_train, y_train)
    print("CV:", search.best_score_)
    return search.best_estimator_


def predict_next_day(model: XGBClassifier, x_last: pd.DataFrame) -> int:
    return int(model.predict(x_last)[0])


def accuracyScore(
    model: XGBClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def get_df(_ticker: str) -> tuple[pd.DataFrame, yf.Ticker] | None:
    ticker = yf.Ticker(_ticker)
    df = ticker.history(period="5y")

    if df.empty:
        return None

    return df, ticker


models_dir = Path("models")
today = date.today().isoformat()


def run_model(
    df: pd.DataFrame, ticker: yf.Ticker
) -> tuple[XGBClassifier, pd.DataFrame, pd.Series]:
    df = create_features(df)
    df = create_labels(df)

    x_train, x_test, y_train, y_test = split_data(df)
    model = train_model(x_train, y_train)
    model_data = (model, x_test, y_test)

    models_dir.mkdir(exist_ok=True)

    with Path(models_dir / f"model_{ticker.ticker}_{today}.pkl").open("wb") as f:
        pickle.dump(model_data, f)

    return model_data


class InvalidTickerError(Exception):
    """Raised when a ticker is invalid or no data is available."""

    pass


def get_df_and_model(
    _ticker: str, force: bool = False
) -> tuple[
    tuple[pd.DataFrame, yf.Ticker], XGBClassifier, tuple[pd.DataFrame, pd.Series]
]:
    test = get_df(_ticker)
    if test is None:
        raise InvalidTickerError(f"Invalid ticker: {_ticker}.")
    df, ticker = test

    path = Path(models_dir / f"model_{ticker.ticker}_{today}.pkl")

    if not path.exists() or force:
        model, x_test, y_test = run_model(df, ticker)
    else:
        with path.open("rb") as f:
            model, x_test, y_test = pickle.load(f)  # noqa: S301

    return (df, ticker), model, (x_test, y_test)
