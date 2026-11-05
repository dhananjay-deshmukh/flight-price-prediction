from pathlib import Path
import re

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def engineer_features(df):
    """Create features without using the target price or row ID."""
    X = df.drop(columns=["id"], errors="ignore").copy()

    # Flight identifiers contain an airline code and flight number.
    flight_parts = X["flight"].astype(str).str.extract(
        r"^\s*([A-Za-z]+)\s*(\d+)\s*$"
    )
    X["flight_code"] = flight_parts[0].fillna("UNKNOWN")
    X["flight_number"] = pd.to_numeric(
        flight_parts[1], errors="coerce"
    )

    
    # Route-specific pricing patterns.
    X["route"] = (
        X["source_city"].astype(str)
        + "_to_"
        + X["destination_city"].astype(str)
    )

    X["route_class"] = (
        X["route"].astype(str)
        + "_"
        + X["flight_class"].astype(str)
    )

    X["duration_per_day_left"] = (
        pd.to_numeric(X["duration"], errors="coerce")
        / pd.to_numeric(X["days_left"], errors="coerce").clip(lower=1)
    )

    # Duration per day until departure is not used:
    # instead retain duration and days_left as separate predictors.
    X = X.drop(columns=["flight"])

    categorical = [
        "airline",
        "route_class",
        "flight_code",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "flight_class",
        "route",
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

    return X, categorical, numeric


def make_model(categorical, numeric):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric,
            ),
        ]
    )

    model = ExtraTreesRegressor(
    n_estimators=400,
    max_features=0.9,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    y = np.log1p(train["price"])
    X, categorical, numeric = engineer_features(
        train.drop(columns=["price"])
    )
    X_test, _, _ = engineer_features(test)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    # First fit: evaluate on rows not used for training.
    validation_model = make_model(categorical, numeric)
    validation_model.fit(X_train, y_train)
    valid_preds = np.expm1(validation_model.predict(X_valid))
    y_valid = np.expm1(y_valid)

    print("\n--- Extra Trees v1: validation results ---")
    print(f"MAE:  {mean_absolute_error(y_valid, valid_preds):.5f}")
    print(
        f"RMSE: "
        f"{np.sqrt(mean_squared_error(y_valid, valid_preds)):.5f}"
    )
    print(f"R2:   {r2_score(y_valid, valid_preds):.5f}")

    # Second fit: train on all labelled data for the submission.
    final_model = make_model(categorical, numeric)
    final_model.fit(X, y)
    predictions = np.expm1(final_model.predict(X_test))
    predictions = np.maximum(predictions, 0)

    submission = pd.DataFrame({
        "id": test["id"],
        "price": predictions,
    })

    # Prices are continuous predictions; preserve precision for now.
    output_path = OUTPUTS / "submission_muskan_extra_trees_v5_log.csv"
    submission.to_csv(output_path, index=False)

    print("\n--- Submission created ---")
    print(f"Path: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())
    print(f"Missing predictions: {submission['price'].isna().sum()}")


if __name__ == "__main__":
    main()
