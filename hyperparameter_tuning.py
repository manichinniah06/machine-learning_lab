"""Hyperparameter tuning with RandomizedSearchCV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor


RANDOM_STATE = 42


@dataclass
class TuningArtifacts:
    """Container with best estimators and search objects."""

    best_models: Dict[str, Pipeline]
    searches: Dict[str, RandomizedSearchCV]


def _run_search(
    model,
    param_distributions: Dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols,
    numerical_cols,
    random_state: int = RANDOM_STATE,
    n_iter: int = 25,
    cv: int = 5,
) -> RandomizedSearchCV:
    preprocessor = build_preprocessor(categorical_cols, numerical_cols)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_weighted",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def tune_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical_cols,
    numerical_cols,
    random_state: int = RANDOM_STATE,
    n_iter: int = 25,
    cv: int = 5,
) -> TuningArtifacts:
    """Tune Random Forest, SVM, and Decision Tree models."""
    rf_params = {
        "model__n_estimators": [100, 200, 400, 600, 800],
        "model__max_depth": [None, 5, 10, 15, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    svm_params = {
        "model__C": np.logspace(-2, 2, 20),
        "model__kernel": ["linear", "rbf", "poly"],
        "model__gamma": ["scale", "auto"],
    }
    dt_params = {
        "model__max_depth": [None, 3, 5, 8, 12, 20],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__criterion": ["gini", "entropy"],
    }

    rf_search = _run_search(
        model=RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_distributions=rf_params,
        X_train=X_train,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        random_state=random_state,
        n_iter=n_iter,
        cv=cv,
    )

    svm_search = _run_search(
        model=SVC(probability=True, random_state=random_state),
        param_distributions=svm_params,
        X_train=X_train,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        random_state=random_state,
        n_iter=n_iter,
        cv=cv,
    )

    dt_search = _run_search(
        model=DecisionTreeClassifier(random_state=random_state),
        param_distributions=dt_params,
        X_train=X_train,
        y_train=y_train,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        random_state=random_state,
        n_iter=n_iter,
        cv=cv,
    )

    searches = {
        "Random Forest": rf_search,
        "SVM": svm_search,
        "Decision Tree": dt_search,
    }

    best_models = {name: search.best_estimator_ for name, search in searches.items()}
    return TuningArtifacts(best_models=best_models, searches=searches) # type: ignore


def best_params_table(searches: Dict[str, RandomizedSearchCV]) -> pd.DataFrame:
    """Create a compact table of best hyperparameters and CV scores."""
    rows = []
    for model_name, search in searches.items():
        rows.append(
            {
                "Model": model_name,
                "Best CV Score": search.best_score_,
                "Best Params": search.best_params_,
            }
        )
    return pd.DataFrame(rows)
