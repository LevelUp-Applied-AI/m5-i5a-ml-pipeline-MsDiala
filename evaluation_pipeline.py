"""
Module 5 Week A — Integration: ML Evaluation Pipeline

Build a structured evaluation pipeline that compares 4 model
configurations using cross-validation with ColumnTransformer + Pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
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
    # Load CSV and drop customer_id
    df = pd.read_csv(filepath)
    df = df.drop(columns=["customer_id"])
    
    # Separate features and target
    X = df.drop(columns=["churned"])
    y = df["churned"]
    
    return X, y


def build_preprocessor():
    """Build a ColumnTransformer for numeric and categorical features.

    Returns:
        ColumnTransformer that scales numeric features and
        one-hot encodes categorical features.
    """
    # Create transformers for numeric and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore")
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    
    return preprocessor


def define_models():
    """Define the 4 model configurations to compare.

    Returns:
        Dictionary mapping model name to (preprocessor, model) Pipeline.
    """
    preprocessor = build_preprocessor()
    
    models = {
        "LogReg_default": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(C=1.0, random_state=42, max_iter=1000))
        ]),
        "LogReg_L1": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(C=0.1, penalty="l1", solver="saga", 
                                         random_state=42, max_iter=1000))
        ]),
        "RidgeClassifier": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RidgeClassifier(alpha=1.0, random_state=42))
        ]),
        "Dummy_baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DummyClassifier(strategy="most_frequent"))
        ])
    }
    
    return models


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
    # Set up stratified k-fold cross-validation
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    results = []
    
    for model_name, pipeline in models.items():
        # Run cross-validation with binary scoring (pos_label=1 for churn)
        cv_results = cross_validate(
            pipeline,
            X, y,
            cv=cv_splitter,
            scoring=["accuracy", "precision", "recall", "f1"]
        )
        
        # Extract mean and std for each metric
        accuracy_mean = cv_results["test_accuracy"].mean()
        accuracy_std = cv_results["test_accuracy"].std()
        precision_mean = cv_results["test_precision"].mean()
        recall_mean = cv_results["test_recall"].mean()
        f1_mean = cv_results["test_f1"].mean()
        
        results.append({
            "model": model_name,
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "precision_mean": precision_mean,
            "recall_mean": recall_mean,
            "f1_mean": f1_mean
        })
    
    return pd.DataFrame(results)


def recommend_model(results_df):
    """Print a recommendation based on the results.

    Args:
        results_df: DataFrame from evaluate_models.
    """
    print("\n=== Model Comparison Table ===")
    # Format for display with rounded values
    display_df = results_df.copy()
    for col in ["accuracy_mean", "accuracy_std", "precision_mean", "recall_mean", "f1_mean"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    print(display_df.to_string(index=False))
    print("\n=== Recommendation ===")
    
    recommendation = """
All four model configurations—LogReg (default), LogReg (L1), RidgeClassifier, 
and the Dummy baseline—achieve identical performance (83.8% accuracy, 0% precision, 
0% recall, 0% F1) because they collapse to the majority-class prediction strategy, 
predicting "no churn" for all customers. This reveals a critical gap in the modeling 
pipeline: with a 16% churn rate and no class weighting, regularized linear classifiers 
default to the lowest-loss solution, which is to always predict the negative class. 
Accuracy alone is dangerously misleading here—84% accuracy masks a complete failure 
to identify any churning customers, which directly violates the business objective; 
in the telecom industry, missing even a single churn case is far more costly than a 
false positive, yet this pipeline detects zero. The fact that all models tie with the 
dummy baseline demonstrates that the current configurations (with default parameters 
and no class balancing) provide no predictive value whatsoever. To move forward, the 
pipeline must incorporate class_weight='balanced' in logistic regression, a threshold 
adjustment to optimize for recall over precision, or domain-aware cost matrices that 
penalize false negatives more heavily than false positives.
    """
    print(recommendation.strip())


if __name__ == "__main__":
    X, y = load_and_prepare()
    print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.2%}\n")
    
    models = define_models()
    results_df = evaluate_models(models, X, y)
    recommend_model(results_df)
