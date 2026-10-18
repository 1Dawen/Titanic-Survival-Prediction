import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def titanic_rows():
    """Fictional Titanic-format records. Not competition data."""
    rows = []
    rng = np.random.default_rng(24)
    titles = ["Mr", "Mrs", "Miss", "Master", "Dr"]
    for i in range(45):
        title = titles[i % len(titles)]
        sex = "female" if title in {"Mrs", "Miss"} else "male"
        rows.append({
            "PassengerId": i + 1,
            "Survived": int(sex == "female" or (i % 13 == 0)),
            "Name": f"Fictional{i}, {title}. Passenger{i}",
            "Pclass": (i % 3) + 1,
            "Sex": sex,
            "Age": float(rng.integers(4, 78)) if i % 5 else np.nan,
            "SibSp": i % 2,
            "Parch": i % 3,
            "Ticket": f"T-{i // 3:03d}",
            "Fare": float(4 + i) if i % 11 else np.nan,
            "Cabin": None,
            "Embarked": ["S", "C", "Q"][i % 3] if i % 9 else None,
        })
    return pd.DataFrame(rows)
