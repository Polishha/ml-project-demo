"""Модуль для трансформации признаков."""

import pandas as pd


def fill_missing(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """Заполняет пропуски в числовых колонках.

    Args:
        df: Исходный DataFrame.
        strategy: Стратегия заполнения ('mean', 'median', 'zero').

    Returns:
        DataFrame без пропусков.

    Raises:
        ValueError: Если передана неизвестная стратегия.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns

    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif strategy == "zero":
        df[numeric_cols] = df[numeric_cols].fillna(0)
    else:
        raise ValueError(f"Неизвестная стратегия: {strategy}")

    return df
