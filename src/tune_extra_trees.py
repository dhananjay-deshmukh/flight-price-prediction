
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from muskan_extra_trees_v1 import engineer_features

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "train.csv")

y = df["price"]
X, categorical, numeric = engineer_features(df.drop(columns=["price"]))

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42
)

configs = [
    {"min_samples_leaf": 1, "max_features": 0.9},
    {"min_samples_leaf": 2, "max_features": 0.9},
    {"min_samples_leaf": 5, "max_features": 0.9},
    {"min_samples_leaf": 10, "max_features": 0.9},
    {"min_samples_leaf": 2, "max_features": 0.7},
]

results = []

for config in configs:
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(
            handle_unknown="ignore", min_frequency=2
        ), categorical),
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("model", ExtraTreesRegressor(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            **config,
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    results.append({
        **config,
        "MAE": mean_absolute_error(y_valid, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_valid, predictions)),
        "R2": r2_score(y_valid, predictions),
    })

print(pd.DataFrame(results).sort_values("MAE").to_string(index=False))
