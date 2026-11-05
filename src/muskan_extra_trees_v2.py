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

    # Duration per day until departure is not used:
    # instead retain duration and days_left as separate predictors.
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
        n_estimators=600,
        max_features=0.9,
        min_samples_leaf=1,
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

    y = train["price"]
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

    
    # First fit: train a separate model for each flight class.
    valid_preds = np.zeros(len(X_valid))

    for flight_class in X_train["flight_class"].unique():
        train_mask = X_train["flight_class"] == flight_class
        valid_mask = X_valid["flight_class"] == flight_class

        model = make_model(categorical, numeric)
        model.fit(X_train.loc[train_mask], y_train.loc[train_mask])
        valid_preds[valid_mask.to_numpy()] = model.predict(
            X_valid.loc[valid_mask]
        )

        print(
            f"{flight_class}: "
            f"{train_mask.sum()} training rows, "
            f"{valid_mask.sum()} validation rows"
        )

    print("\n--- Extra Trees v2: class-specific validation ---")
    print(f"MAE:  {mean_absolute_error(y_valid, valid_preds):.5f}")
    print(
        f"RMSE: "
        f"{np.sqrt(mean_squared_error(y_valid, valid_preds)):.5f}"
    )
    print(f"R2:   {r2_score(y_valid, valid_preds):.5f}")

    # Second fit: train one model per class on all labelled data.
    predictions = np.zeros(len(test))

    for flight_class in X["flight_class"].unique():
        train_mask = X["flight_class"] == flight_class
        test_mask = X_test["flight_class"] == flight_class

        model = make_model(categorical, numeric)
        model.fit(X.loc[train_mask], y.loc[train_mask])
        predictions[test_mask.to_numpy()] = model.predict(
            X_test.loc[test_mask]
        )

    submission = pd.DataFrame({
        "id": test["id"],
        "price": predictions,
    })

    output_path = OUTPUTS / "submission_muskan_extra_trees_v2.csv"
    submission.to_csv(output_path, index=False)

    print("\n--- Submission created ---")
    print(f"Path: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head())
    print(f"Missing predictions: {submission['price'].isna().sum()}")



if __name__ == "__main__":
    main()
