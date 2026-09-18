"""
Run this script after training to produce the final submission CSV.
Usage:
    python src/generate_submission.py --model outputs/model.pkl --tag v1
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Mirror exactly the preprocessing applied in 02_modeling.ipynb.
    # Fill in once feature engineering is finalised.
    raise NotImplementedError("Implement preprocessing to match 02_modeling.ipynb")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to pickled model/pipeline")
    parser.add_argument("--tag", default="v1", help="Version tag for output filename")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    test = pd.read_csv(root / "data" / "test.csv")
    ids = test["id"]

    X = preprocess(test.drop(columns=["id"]))
    model = load_model(args.model)
    preds = model.predict(X)

    out = pd.DataFrame({"id": ids, "price": preds.astype(int)})
    assert out.shape == (4520, 2), f"Unexpected submission shape: {out.shape}"
    assert list(out.columns) == ["id", "price"]

    out_path = root / "outputs" / f"submission_{args.tag}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  shape={out.shape}")


if __name__ == "__main__":
    main()
