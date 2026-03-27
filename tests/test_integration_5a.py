"""Autograder tests for Integration 5A — ML Evaluation Pipeline."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter"))

from evaluation_pipeline import (load_and_prepare, build_preprocessor,
                                  define_models, evaluate_models)


def test_data_loaded():
    result = load_and_prepare(
        os.path.join(os.path.dirname(__file__), "..", "starter", "data", "telecom_churn.csv")
    )
    assert result is not None, "load_and_prepare returned None"
    X, y = result
    assert X.shape[0] > 1000
    assert "churned" not in X.columns, "Target should not be in features"


def test_preprocessor_built():
    prep = build_preprocessor()
    assert prep is not None, "build_preprocessor returned None"
    assert hasattr(prep, "fit_transform"), "Preprocessor must have fit_transform"


def test_models_defined():
    models = define_models()
    assert models is not None, "define_models returned None"
    assert len(models) >= 4, f"Expected 4 models, got {len(models)}"
    for name, pipe in models.items():
        assert hasattr(pipe, "fit"), f"Model '{name}' must have fit method"


def test_evaluation_runs():
    result = load_and_prepare(
        os.path.join(os.path.dirname(__file__), "..", "starter", "data", "telecom_churn.csv")
    )
    assert result is not None
    X, y = result
    models = define_models()
    assert models is not None

    results_df = evaluate_models(models, X, y)
    assert results_df is not None, "evaluate_models returned None"
    assert len(results_df) >= 4, f"Expected 4 rows, got {len(results_df)}"
    assert "accuracy_mean" in results_df.columns, "Missing accuracy_mean column"
