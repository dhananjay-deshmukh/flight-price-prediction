"""
Generate a submission CSV from a saved pipeline.

Usage:
    python src/generate_submission.py --model outputs/model_rf_v1.pkl --tag rf_v1
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

STOPS_MAP = {"zero": 0, "one": 1, "two_or_more": 2}
TIME_MAP  = {"Early_Morning": 0, "Morning": 1, "Afternoon": 2,
             "Evening": 3, "Night": 4, "Late_Night": 5}
CLASS_MAP = {"Economy": 0, "Business": 1}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same ordinal encoding used in 03_random_forest.ipynb.
    The pipeline itself handles passthrough + OHE; this step prepares the
    ordinal columns and drops the high-cardinality flight column."""
    return df.assign(
        stops          = df["stops"].map(STOPS_MAP),
        departure_time = df["departure_time"].map(TIME_MAP),
        arrival_time   = df["arrival_time"].map(TIME_MAP),
        flight_class   = df["flight_class"].map(CLASS_MAP),
    ).drop(columns=["flight"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to pickled pipeline")
    parser.add_argument("--tag", default="v1", help="Version tag for output filename")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    test = pd.read_csv(root / "data" / "test.csv")
    ids  = test["id"]

    X    = preprocess(test.drop(columns=["id"]))
    with open(args.model, "rb") as f:
        pipeline = pickle.load(f)

    preds = np.clip(np.round(pipeline.predict(X)), 1, None).astype(int)

    out = pd.DataFrame({"id": ids, "price": preds})
    assert out.shape == (4520, 2), f"Unexpected shape: {out.shape}"
    assert list(out.columns) == ["id", "price"]
    assert out["price"].isna().sum() == 0

    out_path = root / "outputs" / f"submission_{args.tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  shape={out.shape}  price range: {preds.min()}–{preds.max()}")


if __name__ == "__main__":
    main()
