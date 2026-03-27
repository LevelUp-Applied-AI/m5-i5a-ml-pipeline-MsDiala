"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 4 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.dummy import DummyClassifier


NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents"]

CATEGORICAL_FEATURES = ["gender", "contract_type", "internet_service",
                        "payment_method"]


def load_and_prepare(filepath="data/telecom_churn.csv"):
    """Load data and separate features from target.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features
        and y is a Series of the target (churned).
    """
    # TODO: Load CSV, drop customer_id, separate features and target
    pass


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    # TODO: Create a ColumnTransformer with StandardScaler for numeric
    #       and OneHotEncoder for categorical columns
    pass


def define_models():
    """Define the 4 model configurations to compare.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    # TODO: Create 4 Pipelines, each using the preprocessor + a model:
    #   1. "LogReg_default" — LogisticRegression with default C
    #   2. "LogReg_L1" — LogisticRegression with C=0.1, penalty='l1', solver='saga'
    #   3. "RidgeClassifier" — RidgeClassifier
    #   4. "Dummy_baseline" — DummyClassifier(strategy='most_frequent')
    pass


def evaluate_models(models, X, y, cv=5, random_state=42):
    """Run cross-validation on all models and return results.

    Args:
        models: Dictionary of {name: Pipeline}.
        X: Feature DataFrame.
        y: Target Series.
        cv: Number of folds.
        random_state: Random seed.

    Returns:
        DataFrame with columns: model, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, f1_mean.
    """
    # TODO: Loop over models, run cross_validate with scoring metrics,
    #       collect results into a DataFrame
    pass


def recommend_model(results_df):
    """Print a recommendation based on the results.

    Args:
        results_df: DataFrame from evaluate_models.
    """
    print("\n=== Model Comparison Table ===")
    print(results_df.to_string(index=False))
    print("\n=== Recommendation ===")
    print("Write your recommendation in the PR description.")


if __name__ == "__main__":
    data = load_and_prepare()
    if data is not None:
        X, y = data
        print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.2%}")

        models = define_models()
        if models:
            results = evaluate_models(models, X, y)
            if results is not None:
                recommend_model(results)
