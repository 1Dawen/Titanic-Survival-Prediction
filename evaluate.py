"""Reproduce GroupKFold CatBoost parameter comparisons from Titanic notebook."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from titanic_survival.pipeline import DEFAULT_GRID, run_group_cv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/cv_results.csv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="One small model to validate the setup")
    args = parser.parse_args()
    if not args.train.is_file():
        parser.error(f"Cannot find {args.train}. Download the Kaggle Titanic training CSV first.")
    grid = {"iterations": [30], "depth": [5], "learning_rate": [0.02],
            "loss_function": ["Logloss"]} if args.quick else DEFAULT_GRID
    results = run_group_cv(pd.read_csv(args.train), param_grid=grid, n_splits=args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"Saved cross-validation results to {args.output}")


if __name__ == "__main__":
    main()
