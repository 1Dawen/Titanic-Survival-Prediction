"""Train CatBoost on the Kaggle Titanic training data and create submission.csv."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from titanic_survival.pipeline import DEFAULT_PARAMS, create_submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/test.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--iterations", type=int, default=DEFAULT_PARAMS["iterations"])
    parser.add_argument("--depth", type=int, default=DEFAULT_PARAMS["depth"])
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_PARAMS["learning_rate"])
    args = parser.parse_args()
    for path in (args.train, args.test):
        if not path.is_file():
            parser.error(f"Cannot find {path}. Download the Kaggle Titanic competition CSVs first.")
    params = {"iterations": args.iterations, "depth": args.depth,
              "learning_rate": args.learning_rate, "loss_function": "Logloss"}
    result = create_submission(pd.read_csv(args.train), pd.read_csv(args.test), parameters=params)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} predictions to {args.output}")
    print("This is an inference output, not a Kaggle leaderboard score.")


if __name__ == "__main__":
    main()
