
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "train.csv")

train, valid = train_test_split(
    df, test_size=0.20, random_state=42
)

print("\n--- Validation price distribution ---")
print(valid["price"].describe().round(2))

print("\n--- Validation prices by class ---")
print(
    valid.groupby("flight_class")["price"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
)

print("\n--- Highest validation prices ---")
print(
    valid.nlargest(15, "price")[
        ["airline", "source_city", "destination_city",
         "flight_class", "duration", "days_left", "price"]
    ].to_string(index=False)
)

print("\n--- Zero-price rows in the full dataset ---")
print("Count:", (df["price"] == 0).sum())
print("Percentage:", round((df["price"] == 0).mean() * 100, 3), "%")
