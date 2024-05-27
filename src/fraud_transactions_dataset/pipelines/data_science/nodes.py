"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.3
"""

import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from fraud_transactions_dataset.pipelines.data_science.utils import (
    generate_classification_report,
    generate_confusion_matrix,
)


def train_model(train_df: pd.DataFrame, target_col: str, random_state: int):
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]

    # clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf = xgb.XGBClassifier()

    clf.fit(X_train, y_train)
    return clf


def validate_model(
    ml_model,
    val_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, plt.Figure]:
    X_val, y_val = val_df.drop(target_col, axis=1), val_df[target_col]
    predictions = ml_model.predict(X_val)

    # Classification Report
    report_df = generate_classification_report(y_val, predictions)
    # Confusion Matrix
    figure = generate_confusion_matrix(y_val, predictions)

    return report_df, figure


def predict(ml_model, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X_test, _ = test_df.drop(target_col, axis=1), test_df[target_col]
    predictions = ml_model.predict(X_test)

    test_df["prediction"] = predictions

    return test_df
