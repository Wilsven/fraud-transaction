"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from fraud_transactions_dataset.pipelines.data_science.utils import (
    generate_classification_report,
    generate_confusion_matrix,
)


def evaluate_model(
    predictions_df: pd.DataFrame, target_col: str
) -> tuple[plt.Figure, pd.DataFrame, plt.Figure]:
    def get_auc(
        labels: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        return fpr, tpr, auc_score

    def get_aucpr(
        labels: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | float]:
        precision, recall, _ = precision_recall_curve(labels, scores)
        aucpr_score = np.trapz(recall, precision)
        return precision, recall, aucpr_score

    def plot_metric(
        ax: plt.Axes,
        x: list | np.ndarray,
        y: list | np.ndarray,
        x_label: str,
        y_label: str,
        plot_label: str,
        style: str = "-",
    ):
        ax.plot(x, y, style, label=plot_label)
        ax.legend()
        ax.set_ylabel(x_label)
        ax.set_xlabel(y_label)

    def prediction_summary(
        labels: np.ndarray,
        predicted_score: np.ndarray,
        info: str,
        plot_baseline: bool = True,
        axes: list[plt.Axes] | None = None,
    ) -> list[plt.Axes]:
        if axes is None:
            axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]

        fpr, tpr, auc_score = get_auc(labels, predicted_score)
        plot_metric(
            axes[0],
            fpr,
            tpr,
            "False positive rate",
            "True positive rate",
            "{} AUC={:.4f}".format(info, auc_score),
        )
        if plot_baseline:
            plot_metric(
                axes[0],
                [0, 1],
                [0, 1],
                "False positive rate",
                "True positive rate",
                "Baseline AUC=0.5",
                "r--",
            )
        precision, recall, aucpr_score = get_aucpr(labels, predicted_score)
        plot_metric(
            axes[1],
            recall,
            precision,
            "Recall",
            "Precision",
            "{} AUCPR={:.4f}".format(info, aucpr_score),
        )
        if plot_baseline:
            thr = sum(labels) / len(labels)
            plot_metric(
                axes[1],
                [0, 1],
                [thr, thr],
                "Recall",
                "Precision",
                "Baseline AUCPR={:.4f}".format(thr),
                "r--",
            )
        plt.show()
        return axes

    y_test = predictions_df[target_col]
    predictions = predictions_df["prediction"]

    # Classification Report
    report_df = generate_classification_report(y_test, predictions)
    # Confusion Matrix
    figure = generate_confusion_matrix(y_test, predictions)

    fig = plt.figure()
    fig.set_figheight(4.5)
    fig.set_figwidth(4.5 * 2)
    axes = prediction_summary(
        y_test.values,
        predictions.values,
        "Fraud Detection",
    )
    return fig, report_df, figure
