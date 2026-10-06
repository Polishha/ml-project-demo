"""Тесты для модуля трансформации."""

import pandas as pd
from src.features.transformer import fill_missing


def test_fill_missing_mean() -> None:
    """Проверяет заполнение пропусков средним."""
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, None]})
    result = fill_missing(df, strategy="mean")
    assert not result.isna().any().any()
    assert result["a"].iloc[1] == 2.0
