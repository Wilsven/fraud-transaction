"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_transactions_dataset.pipelines.data_science.nodes import (  # predict,
    predict,
    train_model,
    validate_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=[
                    "fraud_train_processed",
                    "params:target_col",
                    "params:random_state",
                ],
                outputs="ml_model",
                name="train_model_node",
            ),
            node(
                func=validate_model,
                inputs=["ml_model", "fraud_val_processed", "params:target_col"],
                outputs=["classification_val_report", "confusion_val_matrix"],
                name="validate_model_node",
            ),
            node(
                func=predict,
                inputs=["ml_model", "fraud_test_processed", "params:target_col"],
                outputs="predictions",
                name="predict_node",
            ),
        ]
    )
