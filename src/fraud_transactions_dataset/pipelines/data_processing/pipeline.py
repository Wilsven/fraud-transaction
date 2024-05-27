"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_transactions_dataset.pipelines.data_processing.nodes import (
    merge_raw_data, prepare_data, preprocess_data, train_test_val_split)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=merge_raw_data,
                inputs=["fraud_train_raw", "fraud_test_raw", "params:predictor_cols"],
                outputs="raw_merged",
                name="merge_raw_data_node",
            ),
            node(
                func=prepare_data,
                inputs=["raw_merged", "params:top_categories"],
                outputs="raw_merged_prepared",
                name="prepare_data_node",
            ),
            node(
                func=train_test_val_split,
                inputs=[
                    "raw_merged_prepared",
                    "params:target_col",
                    "params:test_size",
                    "params:random_state",
                    "params:undersampling",
                ],
                outputs=[
                    "fraud_train_unprocessed",
                    "fraud_test_unprocessed",
                    "fraud_val_unprocessed",
                ],
                name="train_test_val_split_node",
            ),
            node(
                func=preprocess_data,
                inputs=[
                    "fraud_train_unprocessed",
                    "fraud_test_unprocessed",
                    "fraud_val_unprocessed",
                    "params:target_col",
                    "params:preprocess_features.numeric_features",
                    "params:preprocess_features.categorical_features",
                ],
                outputs=[
                    "fraud_train_processed",
                    "fraud_test_processed",
                    "fraud_val_processed",
                ],
                name="preprocess_data_node",
            ),
        ]
    )
