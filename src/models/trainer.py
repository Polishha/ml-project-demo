"""Модуль для обучения модели."""

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass
class TrainResult:
    """Результат обучения модели."""

    model: LogisticRegression
    accuracy: float


def train_logistic_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> TrainResult:
    """Обучает логистическую регрессию и возвращает метрику.

    Args:
        x_train: Признаки для обучения.
        y_train: Целевая переменная для обучения.
        x_test: Признаки для теста.
        y_test: Целевая переменная для теста.

    Returns:
        TrainResult с моделью и accuracy.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    return TrainResult(model=model, accuracy=accuracy)
