from __future__ import annotations

import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATA_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    TARGET,
    RANDOM_STATE,
    TEST_SIZE
)

from transformers import IQRClipper


def load_data() -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    dataframe = pd.read_csv(DATA_PATH)

    return dataframe


def prepare_features(
    dataframe: pd.DataFrame
):
    """
    Separate features and target.
    """

    if TARGET not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    dataframe = dataframe.copy()

    # Loan_ID is only an identifier, so remove it
    if "Loan_ID" in dataframe.columns:
        dataframe = dataframe.drop(
            columns=["Loan_ID"]
        )

    X = dataframe.drop(
        columns=[TARGET]
    )

    y = dataframe[TARGET].map({
        "Y": 1,
        "N": 0
    })

    if y.isnull().any():
        raise ValueError(
            "Loan_Status contains unexpected values."
        )

    return X, y


def get_feature_columns(
    X: pd.DataFrame
):
    """
    Identify numerical and categorical columns.
    """

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


def build_pipeline(
    numerical_columns,
    categorical_columns
):
    """
    Create preprocessing and Random Forest pipeline.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "iqr_clipper",
                IQRClipper(
                    factor=1.5
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline


def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate trained model.
    """

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    confusion = confusion_matrix(
        y_test,
        predictions
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Rejected",
            "Approved"
        ],
        zero_division=0
    )

    metrics = {
        "accuracy": round(
            accuracy,
            4
        ),
        "precision": round(
            precision,
            4
        ),
        "recall": round(
            recall,
            4
        ),
        "f1_score": round(
            f1,
            4
        ),
        "roc_auc": round(
            roc_auc,
            4
        ),
        "confusion_matrix": (
            confusion.tolist()
        )
    }

    print("\n========================================")
    print("          MODEL EVALUATION")
    print("========================================")

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nClassification Report:")
    print(report)

    return metrics


def save_model(
    model
):
    """
    Save trained model pipeline.
    """

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved successfully:\n{MODEL_PATH}"
    )


def save_metrics(
    metrics
):
    """
    Save evaluation metrics as JSON.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_path = (
        OUTPUT_DIR /
        "model_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"\nMetrics saved successfully:\n{metrics_path}"
    )


def train_model():
    """
    Run complete model training process.
    """

    print("\n========================================")
    print("       LOAN MODEL TRAINING STARTED")
    print("========================================")

    dataframe = load_data()

    X, y = prepare_features(
        dataframe
    )

    numerical_columns, categorical_columns = (
        get_feature_columns(X)
    )

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )
    )

    print(
        f"\nTraining samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    pipeline = build_pipeline(
        numerical_columns,
        categorical_columns
    )

    print("\nTraining Random Forest model...")

    pipeline.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    metrics = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    save_model(
        pipeline
    )

    save_metrics(
        metrics
    )

    print("\n========================================")
    print("       TRAINING COMPLETED SUCCESSFULLY")
    print("========================================")

    return pipeline


if __name__ == "__main__":
    train_model()