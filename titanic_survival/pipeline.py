"""CatBoost Titanic classifier with ticket-grouped validation and fold-safe imputation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, ParameterGrid

CAT_FEATURES = ["Sex", "Embarked", "Title"]
FEATURES = ["Pclass", "Sex", "Age", "Embarked", "Title", "FamilySize", "AgeClass", "Fare"]
REQUIRED = {"PassengerId", "Name", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Ticket"}
TITLE_ALIASES = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Col": "Rare",
    "Major": "Rare", "Capt": "Rare", "Sir": "Rare", "Lady": "Rare",
    "the Countess": "Rare", "Jonkheer": "Rare", "Don": "Rare",
    "Dona": "Rare", "Rev": "Rare", "Dr": "Rare",
}
DEFAULT_GRID = {
    "iterations": [500, 1000],
    "depth": [5, 10],
    "learning_rate": [0.02, 0.05],
    "loss_function": ["Logloss"],
}
DEFAULT_PARAMS = {"depth": 5, "iterations": 500, "learning_rate": 0.02, "loss_function": "Logloss"}


@dataclass(frozen=True)
class PreprocessingStats:
    age_by_title: pd.Series
    fallback_age: float
    fallback_fare: float


def _check_raw(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED.difference(df.columns))
    if missing:
        raise ValueError(f"Missing Titanic columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Input dataframe is empty")


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract titles and family attributes without learning dataset statistics."""
    _check_raw(raw)
    df = raw.copy()
    titles = df["Name"].fillna("").astype(str).str.extract(r",\s*([^.]+)\.", expand=False)
    df["Title"] = titles.fillna("Unknown").str.strip().replace(TITLE_ALIASES)
    df["FamilySize"] = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0) + pd.to_numeric(df["Parch"], errors="coerce").fillna(0) + 1
    df["IsAlone"] = df["FamilySize"].eq(1).astype(int)
    return df


def fit_preprocessing(train_engineered: pd.DataFrame) -> PreprocessingStats:
    """Fit age and fare imputation ONLY on training observations."""
    age = pd.to_numeric(train_engineered["Age"], errors="coerce")
    fare = pd.to_numeric(train_engineered["Fare"], errors="coerce")
    if not age.notna().any() or not fare.notna().any():
        raise ValueError("Training data must contain at least one valid Age and Fare")
    by_title = age.groupby(train_engineered["Title"]).median()
    return PreprocessingStats(age_by_title=by_title, fallback_age=float(age.median()), fallback_fare=float(fare.median()))


def transform_features(engineered: pd.DataFrame, stats: PreprocessingStats) -> pd.DataFrame:
    """Transform training/validation/test with training-only fitted statistics."""
    x = engineered.copy()
    x["Pclass"] = pd.to_numeric(x["Pclass"], errors="coerce")
    x["Age"] = pd.to_numeric(x["Age"], errors="coerce").fillna(x["Title"].map(stats.age_by_title)).fillna(stats.fallback_age)
    x["Fare"] = pd.to_numeric(x["Fare"], errors="coerce").fillna(stats.fallback_fare)
    x["AgeClass"] = x["Age"] * x["Pclass"]
    x["Embarked"] = x["Embarked"].fillna("S")
    for col in CAT_FEATURES:
        x[col] = x[col].fillna("Unknown").astype(str)
    features = x[FEATURES].copy()
    if features.isna().any().any():
        raise ValueError("Unresolved missing values remain in model features")
    return features


def make_classifier(parameters: Mapping | None = None, *, random_state: int = 24) -> CatBoostClassifier:
    params = dict(DEFAULT_PARAMS if parameters is None else parameters)
    return CatBoostClassifier(**params, random_seed=random_state, verbose=False, allow_writing_files=False, thread_count=2)


def run_group_cv(raw_train: pd.DataFrame, *, param_grid: Mapping | None = None,
                 n_splits: int = 5, random_state: int = 24) -> pd.DataFrame:
    """Compare CatBoost settings with GroupKFold grouped by shared tickets."""
    if "Survived" not in raw_train:
        raise ValueError("Training CSV needs the Survived target")
    engineered = engineer_features(raw_train)
    groups = engineered["Ticket"].fillna("Unknown").astype(str)
    if groups.nunique() < n_splits:
        raise ValueError("Not enough distinct ticket groups for GroupKFold")
    y = pd.to_numeric(engineered["Survived"], errors="raise").astype(int)
    records = []
    cv = GroupKFold(n_splits=n_splits)
    grid = DEFAULT_GRID if param_grid is None else param_grid
    for params in ParameterGrid(grid):
        scores = []
        for train_idx, valid_idx in cv.split(engineered, y, groups=groups):
            fold_train = engineered.iloc[train_idx].copy()
            fold_valid = engineered.iloc[valid_idx].copy()
            stats = fit_preprocessing(fold_train)
            x_train = transform_features(fold_train, stats)
            x_valid = transform_features(fold_valid, stats)
            model = make_classifier(params, random_state=random_state)
            model.fit(x_train, y.iloc[train_idx], cat_features=CAT_FEATURES)
            prediction = np.asarray(model.predict(x_valid)).astype(int).ravel()
            scores.append(float(accuracy_score(y.iloc[valid_idx], prediction)))
        records.append({**params, "mean_accuracy": float(np.mean(scores)),
                        "std_accuracy": float(np.std(scores)), "folds": len(scores)})
    return pd.DataFrame(records).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)


def create_submission(raw_train: pd.DataFrame, raw_test: pd.DataFrame,
                      *, parameters: Mapping | None = None, random_state: int = 24) -> pd.DataFrame:
    """Train on labeled Kaggle records and predict the Kaggle test CSV."""
    if "Survived" not in raw_train:
        raise ValueError("Training CSV needs the Survived target")
    train = engineer_features(raw_train)
    test = engineer_features(raw_test)
    stats = fit_preprocessing(train)
    x_train, x_test = transform_features(train, stats), transform_features(test, stats)
    labels = pd.to_numeric(train["Survived"], errors="raise").astype(int)
    model = make_classifier(parameters, random_state=random_state)
    model.fit(x_train, labels, cat_features=CAT_FEATURES)
    predicted = np.asarray(model.predict(x_test)).astype(int).ravel()
    return pd.DataFrame({"PassengerId": test["PassengerId"].to_numpy(), "Survived": predicted})
