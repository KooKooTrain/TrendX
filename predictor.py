import pathlib
import pickle
from collections import Counter

import yfinance as yf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def create_features(df, target_col="Close"):
    df = df.copy()
    windows = [3, 5, 10, 20, 50]
    for w in windows:
        df[f"roll_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"roll_std_{w}"] = df[target_col].rolling(w).std()
        df[f"roll_min_{w}"] = df[target_col].rolling(w).min()
        df[f"roll_max_{w}"] = df[target_col].rolling(w).max()
        df[f"pct_change_{w}"] = df[target_col].pct_change(w)
        df[f"momentum_{w}"] = df[target_col] - df[target_col].shift(w)
        df[f"vol_mean_{w}"] = df["Volume"].rolling(w).mean()
        df[f"vol_std_{w}"] = df["Volume"].rolling(w).std()

    for lag in range(1, 6):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    df = df.dropna()
    return df


def create_labels(df, target_col="Close"):
    df = df.copy()
    df["target"] = (df[target_col].shift(-1) > df[target_col]).astype(int)
    return df.dropna()


def scale_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def split_data(df):
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)


def train_model(x_train, y_train):
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    param_distributions = {
        "n_estimators": [400, 500, 600],
        "learning_rate": [0.01, 0.03, 0.05],
        "max_depth": [3, 4, 5, 6],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=20,
        scoring="accuracy",
        cv=tscv,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def predict_next_day(model, scaler, x_last):
    x_scaled = scaler.transform(x_last)
    return model.predict(x_scaled)[0]


def accuracyScore(model, x_test_scaled, y_test):
    y_pred = model.predict(x_test_scaled)
    return accuracy_score(y_test, y_pred) * 100


df = yf.download("TSLA", period="5y", auto_adjust=True)


def run_model(df):
    df = create_features(df)
    df = create_labels(df)

    x_train, x_test, y_train, y_test = split_data(df)
    x_train_scaled, x_test_scaled, scaler = scale_data(x_train, x_test)
    model = train_model(x_train_scaled, y_train)

    model.save_model("model.ubj")

    with pathlib.Path("./scaler.pkl").open("wb") as f:
        pickle.dump([scaler, x_test_scaled, y_test], f)


if __name__ == "__main__":
    run_model(df)
