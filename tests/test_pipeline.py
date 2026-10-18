import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from titanic_survival.pipeline import (
    CAT_FEATURES, FEATURES, engineer_features, fit_preprocessing,
    transform_features, run_group_cv, create_submission,
)


def test_title_normalization_and_family_size(titanic_rows):
    df = titanic_rows.iloc[:5].copy()
    df.loc[df.index[0], "Name"] = "Surname, Dr. Example"
    df.loc[df.index[1], "Name"] = "Surname, Mme. Example"
    out = engineer_features(df)
    assert out.iloc[0]["Title"] == "Rare"
    assert out.iloc[1]["Title"] == "Mrs"
    assert out.iloc[0]["FamilySize"] == out.iloc[0]["SibSp"] + out.iloc[0]["Parch"] + 1
    assert (out["IsAlone"] == out["FamilySize"].eq(1).astype(int)).all()


def test_missing_raw_columns_are_reported(titanic_rows):
    with pytest.raises(ValueError, match="Ticket"):
        engineer_features(titanic_rows.drop(columns="Ticket"))


def test_empty_input_is_rejected(titanic_rows):
    with pytest.raises(ValueError, match="empty"):
        engineer_features(titanic_rows.iloc[:0])


def test_preprocessing_uses_train_only_medians(titanic_rows):
    train = engineer_features(titanic_rows.iloc[1:30])
    valid = engineer_features(titanic_rows.iloc[30:32].copy())
    valid.loc[valid.index[0], "Age"] = np.nan
    valid.loc[valid.index[0], "Title"] = "Unseen"
    valid.loc[valid.index[0], "Fare"] = np.nan
    stats = fit_preprocessing(train)
    transformed = transform_features(valid, stats)
    assert transformed.iloc[0]["Age"] == stats.fallback_age
    assert transformed.iloc[0]["Fare"] == stats.fallback_fare
    assert transformed.iloc[0]["AgeClass"] == stats.fallback_age * transformed.iloc[0]["Pclass"]
    assert not transformed.isna().any().any()


def test_features_match_group_cv_and_final_model(titanic_rows):
    engineered = engineer_features(titanic_rows)
    x = transform_features(engineered, fit_preprocessing(engineered))
    assert list(x.columns) == FEATURES
    assert set(CAT_FEATURES).issubset(x.columns)
    assert "Fare" in FEATURES
    assert "Ticket" not in FEATURES


def test_ticket_groups_do_not_overlap_between_folds(titanic_rows):
    groups = titanic_rows["Ticket"]
    for tr, va in GroupKFold(n_splits=5).split(titanic_rows, titanic_rows["Survived"], groups):
        assert not set(groups.iloc[tr]) & set(groups.iloc[va])


def test_small_parameter_cv_returns_scores(titanic_rows):
    grid = {"depth": [3], "iterations": [8], "learning_rate": [0.05],
            "loss_function": ["Logloss"]}
    result = run_group_cv(titanic_rows, param_grid=grid, n_splits=3)
    assert len(result) == 1
    assert result.iloc[0]["folds"] == 3
    assert 0 <= result.iloc[0]["mean_accuracy"] <= 1
    assert np.isfinite(result.iloc[0]["std_accuracy"])


def test_submission_has_required_schema(titanic_rows):
    train = titanic_rows.iloc[:36].copy()
    test = titanic_rows.iloc[36:].drop(columns="Survived").copy()
    result = create_submission(train, test, parameters={
        "iterations": 8, "depth": 3, "learning_rate": 0.05,
        "loss_function": "Logloss"})
    assert result.columns.tolist() == ["PassengerId", "Survived"]
    assert result["PassengerId"].tolist() == test["PassengerId"].tolist()
    assert result["Survived"].isin([0, 1]).all()
