
from pathlib import Path
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from catboost_v10 import prepare_features, RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
train = pd.read_csv(ROOT / "data" / "train.csv")

y = train["price"]
X, categorical = prepare_features(train.drop(columns=["price"]))
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
cat_indices = [X.columns.get_loc(c) for c in categorical]

configs = [
    {"depth": 6, "learning_rate": 0.05},
    {"depth": 7, "learning_rate": 0.05},
    {"depth": 8, "learning_rate": 0.03},
    {"depth": 9, "learning_rate": 0.03},
]

results = []

for config in configs:
    print(f"\nTesting {config}", flush=True)

    model = CatBoostRegressor(
        iterations=1800,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        **config,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_indices,
        eval_set=(X_valid, y_valid),
        early_stopping_rounds=150,
        verbose=False,
    )

    pred = model.predict(X_valid)
    result = {
        **config,
        "best_iteration": model.best_iteration_,
        "R2": r2_score(y_valid, pred),
        "MAE": mean_absolute_error(y_valid, pred),
        "RMSE": np.sqrt(mean_squared_error(y_valid, pred)),
    }
    results.append(result)
    print(result, flush=True)

print("\n=== RESULTS (sorted by R2) ===")
print(pd.DataFrame(results).sort_values("R2", ascending=False).to_string(index=False))
