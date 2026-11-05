import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import re

df = pd.read_csv("data/train.csv")

def engineer_features(df):
    X = df.drop(columns=["id", "price"], errors="ignore").copy()

    parsed = X["flight"].astype(str).str.extract(
        r"^\s*([A-Za-z]+)\s*(\d+)\s*$"
    )
    X["flight_code"] = parsed[0].fillna("UNKNOWN")
    X["flight_number"] = pd.to_numeric(parsed[1], errors="coerce")

    X["route"] = X["source_city"] + "_to_" + X["destination_city"]
    X["route_class"] = X["route"] + "_" + X["flight_class"]
    X = X.drop(columns=["flight"])

    categorical = [
        "airline", "route_class", "flight_code", "source_city",
        "departure_time", "stops", "arrival_time", "destination_city",
        "flight_class", "route"
    ]
    numeric = ["duration", "days_left", "flight_number"]

    return X, categorical, numeric

X, categorical, numeric = engineer_features(df)
y = df["price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Exclude zero-price rows from training only.
keep = y_train > 0
X_train = X_train.loc[keep]
y_train = y_train.loc[keep]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), categorical),
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ExtraTreesRegressor(
        n_estimators=400,
        max_features=0.9,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)
pred = model.predict(X_val)

print("\nEXTRA TREES V12: ZERO-PRICE TRAINING ROWS REMOVED")
print("Training rows removed:", int((~keep).sum()))
print("Validation rows:", len(y_val))
print("MAE:", round(mean_absolute_error(y_val, pred), 5))
print("RMSE:", round(np.sqrt(mean_squared_error(y_val, pred)), 5))
print("R2:", round(r2_score(y_val, pred), 5))
