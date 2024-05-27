"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.3
"""

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


def merge_raw_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame, predictor_cols: list[str]
) -> pd.DataFrame:
    # Select only columns in `predictor_cols`
    train_df = train_df[predictor_cols]
    test_df = test_df[predictor_cols]

    # Tag dataset
    train_df["dataset"] = "train"
    test_df["dataset"] = "test"

    merged_df = pd.concat([train_df, test_df], axis=0)

    return merged_df


def prepare_data(merged_df: pd.DataFrame, top_categories: dict) -> pd.DataFrame:
    top_cities = top_categories["cities"]
    top_states = top_categories["states"]

    merged_df["trans_date_trans_time"] = pd.to_datetime(
        merged_df["trans_date_trans_time"]
    )
    merged_df["dob"] = pd.to_datetime(merged_df["dob"])

    # Day of week
    merged_df["dayofweek"] = merged_df["trans_date_trans_time"].dt.dayofweek
    # Hour
    merged_df["hour"] = merged_df["trans_date_trans_time"].dt.hour
    # Age
    merged_df["age"] = (
        merged_df["trans_date_trans_time"].dt.year - merged_df["dob"].dt.year
    )

    # Group categories into `Other` so we don't end up
    # with an overly large and sparse one-hot encoded dataset
    merged_df["city"] = merged_df["city"].apply(
        lambda x: "Other" if x not in top_cities else x
    )
    merged_df["state"] = merged_df["state"].apply(
        lambda x: "Other" if x not in top_states else x
    )

    merged_prepared_df = merged_df.drop(["trans_date_trans_time", "dob"], axis=1)
    return merged_prepared_df


def train_test_val_split(
    merged_prepared_df: pd.DataFrame,
    target_col: str,
    test_size: int,
    random_state: int,
    undersampling_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df_unprocessed = (
        merged_prepared_df[merged_prepared_df["dataset"] == "train"]
        .drop("dataset", axis=1)
        .reset_index(drop=True)
    )
    test_df_unprocessed = (
        merged_prepared_df[merged_prepared_df["dataset"] == "test"]
        .drop("dataset", axis=1)
        .reset_index(drop=True)
    )

    X = train_df_unprocessed.drop(target_col, axis=1)
    y = train_df_unprocessed[target_col]

    desired_proportion = undersampling_params["desired_proportion"]
    total_samples = undersampling_params["total_samples"]
    fraud_samples = int(total_samples * desired_proportion)

    # Create RandomUnderSampler with the desired sampling strategy
    rus = RandomUnderSampler(
        sampling_strategy={0: total_samples - fraud_samples, 1: fraud_samples},
        random_state=random_state,
    )

    # Apply random undersampling to the original dataset
    X_resampled, y_resampled = rus.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, stratify=y_resampled, test_size=test_size
    )

    X_train["is_fraud"] = y_train
    X_val["is_fraud"] = y_val

    train_df_unprocessed = X_train
    val_df_unprocessed = X_val

    return train_df_unprocessed, test_df_unprocessed, val_df_unprocessed


def preprocess_data(
    fraud_train_unprocessed: pd.DataFrame,
    fraud_test_unprocessed: pd.DataFrame,
    fraud_val_unprocessed: pd.DataFrame,
    target_col: str,
    numeric_features: dict,
    categorical_features: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    standard_features = numeric_features["standard"]
    minmax_features = numeric_features["minmax"]
    onehot_features = categorical_features["onehot"]

    X_train, y_train = (
        fraud_train_unprocessed.drop(target_col, axis=1),
        fraud_train_unprocessed[target_col],
    )
    X_test, y_test = (
        fraud_test_unprocessed.drop(target_col, axis=1),
        fraud_test_unprocessed[target_col],
    )
    X_val, y_val = (
        fraud_val_unprocessed.drop(target_col, axis=1),
        fraud_val_unprocessed[target_col],
    )

    standard_transformer = Pipeline(steps=[("scaler", StandardScaler())], verbose=True)
    minmax_transformer = Pipeline(steps=[("minmax", MinMaxScaler())], verbose=True)
    onehot_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))],
        verbose=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_standard", standard_transformer, standard_features),
            ("numeric_minmax", minmax_transformer, minmax_features),
            ("categorical_onehot", onehot_transformer, onehot_features),
        ]
    )

    cat_feats = []
    for feat in onehot_features:
        c = X_train[onehot_features][feat].unique().tolist()
        cat_feats.extend(c)

    processed_train_df = preprocessor.fit_transform(X_train)
    processed_train_df = pd.DataFrame(
        processed_train_df, columns=[*standard_features, *minmax_features, *cat_feats]
    )
    processed_test_df = preprocessor.transform(X_test)
    processed_test_df = pd.DataFrame(
        processed_test_df, columns=[*standard_features, *minmax_features, *cat_feats]
    )
    processed_val_df = preprocessor.transform(X_val)
    processed_val_df = pd.DataFrame(
        processed_val_df, columns=[*standard_features, *minmax_features, *cat_feats]
    )

    processed_train_df["is_fraud"] = y_train.to_list()
    processed_test_df["is_fraud"] = y_test.to_list()
    processed_val_df["is_fraud"] = y_val.to_list()

    return processed_train_df, processed_test_df, processed_val_df
