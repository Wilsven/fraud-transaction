from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def generate_classification_report(
    y_val: pd.Series, predictions: np.ndarray
) -> pd.DataFrame:
    report = classification_report(y_val, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return report_df


def generate_confusion_matrix(y_val: pd.Series, predictions: np.ndarray) -> plt.Figure:
    _, ax = plt.subplots(figsize=(8, 8))
    cm = confusion_matrix(y_val, predictions)

    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_normalized_percentages = cm / np.sum(cm, axis=0, keepdims=True)
    group_normalized_percentages_2 = [
        "{0:.2%}".format(value) for value in group_normalized_percentages.ravel()
    ]
    cell_labels = [
        f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_normalized_percentages_2)
    ]
    cell_labels = np.asarray(cell_labels).reshape(2, 2)
    sns.heatmap(
        100.0 * group_normalized_percentages,
        annot=cell_labels,
        cmap="Blues",
        fmt="",
        ax=ax,
        cbar_kws={"format": "%.0f%%"},
    )

    # Labels, title and ticks
    ax.set_xlabel("Predicted labels", fontsize=14)
    ax.set_ylabel("True labels", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=20)
    ax.xaxis.set_ticklabels(["Not Fraud", "Fraud"])
    ax.yaxis.set_ticklabels(["Not Fraud", "Fraud"])
    plt.tight_layout()

    return ax.figure
