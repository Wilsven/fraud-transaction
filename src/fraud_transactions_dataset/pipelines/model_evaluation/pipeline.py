"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_transactions_dataset.pipelines.model_evaluation.nodes import evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_model,
                inputs=["predictions", "params:target_col"],
                outputs=[
                    "evaluation_plot",
                    "classification_test_report",
                    "confusion_test_matrix",
                ],
                name="evaluate_model_node",
            )
        ]
    )
