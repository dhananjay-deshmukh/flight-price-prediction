
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def prepare_features(df):
    X = df.drop(columns=["id"], errors="ignore").copy()

    # Extract flight code and flight number.
    flight_parts = X["flight"].astype(str).str.extract(
        r"^\s*([A-Za-z]+)\s*(\d+)\s*$"
    )
    X["flight_code"] = flight_parts[0].fillna("UNKNOWN")
    X["flight_number"] = pd.to_numeric(
        flight_parts[1], errors="coerce"
    )

    # Create route features.
    X["route"] = (
        X["source_city"].astype(str)
        + "_to_"
        + X["destination_city"].astype(str)
    )
    X["route_class"] = (
        X["route"] + "_" + X["flight_class"].astype(str)
    )

    # Keep duration and days_left separately in this first experiment.
    X = X.drop(columns=["flight"])

    categorical = [
        "airline",
        "flight_code",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "flight_class",
        "route",
        "route_class",
    ]

    numeric = [
        "duration",
        "days_left",
        "flight_number",
    ]

    for col in categorical:
        X[col] = X[col].fillna("Unknown").astype(str)

    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, categorical


def make_model(categorical):
    return CatBoostRegressor(
        iterations=1200,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        eval_metric="R2",
        random_seed=RANDOM_STATE,
        verbose=200,
        allow_writing_files=False,
        thread_count=-1,
    )


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    y = train["price"]
    X, categorical = prepare_features(train.drop(columns=["price"]))
    X_test, _ = prepare_features(test)

    # CatBoost accepts NaN in numeric features and categorical strings.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    cat_indices = [X.columns.get_loc(col) for col in categorical]

    model = make_model(categorical)
    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=100,
        verbose=200,
    )

    valid_preds = model.predict(X_valid)

    print("\n--- CatBoost v10: validation results ---")
    print(f"MAE:  {mean_absolute_error(y_valid, valid_preds):.5f}")
    print(
        f"RMSE: {np.sqrt(mean_squared_error(y_valid, valid_preds)):.5f}"
    )
    print(f"R2:   {r2_score(y_valid, valid_preds):.5f}")
    print(f"Best iteration: {model.best_iteration_}")

    # Retrain on all labelled data using the selected iteration count.
    final_model = make_model(categorical)
    final_model.set_params(
        iterations=max(1, model.best_iteration_ + 1)
    )
    final_model.fit(
        X,
        y,
        cat_features=cat_indices,
        verbose=200,
    )

    predictions = final_model.predict(X_test)

    submission = pd.DataFrame({
        "id": test["id"],
        "price": predictions,
    })

    output_path = OUTPUTS / "submission_catboost_v10.csv"
    submission.to_csv(output_path, index=False)

    print("\n--- Submission created ---")
    print(f"Path: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())
    print(f"Missing predictions: {submission['price'].isna().sum()}")


if __name__ == "__main__":
    main()
