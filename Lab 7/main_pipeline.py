"""End-to-end ML pipeline orchestration for F1 deviation prediction.

This script runs all stages: data loading, feature engineering, model training,
hyperparameter tuning, evaluation, and disagreement analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import pandas as pd

import disagreement_analysis
import evaluation
import hyperparameter_tuning
import model_training


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_metrics_table(metrics_df: pd.DataFrame) -> None:
    """Pretty-print metrics comparison table."""
    print(metrics_df.to_string(index=False))


def stage_1_load_and_prepare(data_path: str) -> pd.DataFrame:
    """Load raw CSV and prepare for processing."""
    print_section("STAGE 1: Load Dataset")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from: {data_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Class distribution:\n{df['class_label'].value_counts()}\n")
    
    return df


def stage_2_train_baseline_models(data: pd.DataFrame) -> tuple:
    """Train baseline models without hyperparameter tuning."""
    print_section("STAGE 2: Train Baseline Models")
    
    print("Running feature engineering, preprocessing, and training...")
    artifacts = model_training.train_all_models(
        raw_dataframe=data,
        test_size=0.2,
        dnf_strategy="mark",
        rolling_window=5,
    )
    
    print(f"Trained models: {list(artifacts.models.keys())}")
    print(f"Train set shape: {artifacts.X_train.shape}")
    print(f"Test set shape: {artifacts.X_test.shape}")
    print(f"Target classes: {list(artifacts.label_encoder.classes_)}\n")  # type: ignore
    
    return artifacts, artifacts.label_encoder


def stage_3_evaluate_baseline(artifacts: Any, label_encoder: Any) -> tuple:
    """Evaluate baseline models on test set."""
    print_section("STAGE 3: Evaluate Baseline Models")
    
    metrics_df, confusion_matrices, predictions = evaluation.evaluate_models(
        models=artifacts.models,
        X_test=artifacts.X_test,
        y_test=artifacts.y_test,
    )
    
    print_metrics_table(metrics_df)
    evaluation.print_confusion_matrices(
        confusion_matrices,
        class_names=label_encoder.classes_,  # type: ignore
    )
    
    return metrics_df, confusion_matrices, predictions


def stage_4_hyperparameter_tuning(artifacts: Any) -> Any:
    """Perform RandomizedSearchCV for RF, SVM, and Decision Tree."""
    print_section("STAGE 4: Hyperparameter Tuning (RandomizedSearchCV)")
    
    print("Tuning Random Forest, SVM, and Decision Tree...")
    print("(This may take several minutes)\n")
    
    tuning_results = hyperparameter_tuning.tune_models(
        X_train=artifacts.X_train,
        y_train=artifacts.y_train,
        categorical_cols=artifacts.categorical_cols,
        numerical_cols=artifacts.numerical_cols,
        n_iter=25,
        cv=5,
    )
    
    best_params_df = hyperparameter_tuning.best_params_table(tuning_results.searches)
    print("Best hyperparameters found:\n")
    for idx, row in best_params_df.iterrows():
        print(f"Model: {row['Model']}")
        print(f"  Best CV Score: {row['Best CV Score']:.4f}")
        print(f"  Best Params: {row['Best Params']}\n")
    
    return tuning_results


def stage_5_evaluate_tuned_models(tuning_results: Any, artifacts: Any, label_encoder: Any) -> tuple:
    """Evaluate tuned models on test set."""
    print_section("STAGE 5: Evaluate Tuned Models")
    
    metrics_df, confusion_matrices, predictions = evaluation.evaluate_models(
        models=tuning_results.best_models,
        X_test=artifacts.X_test,
        y_test=artifacts.y_test,
    )
    
    print("Tuned Model Metrics:\n")
    print_metrics_table(metrics_df)
    evaluation.print_confusion_matrices(
        confusion_matrices,
        class_names=label_encoder.classes_,  # type: ignore
    )
    
    return metrics_df, confusion_matrices, predictions


def stage_6_disagreement_analysis(artifacts: Any, tuning_results: Any, label_encoder: Any) -> Dict:
    """Analyze model disagreement and uncertainty."""
    print_section("STAGE 6: Model Disagreement Analysis")
    
    disagreement_results = disagreement_analysis.run_disagreement_analysis(
        models=tuning_results.best_models,
        X_test=artifacts.X_test,
        label_encoder=label_encoder,
        grid_col="grid_position",
    )
    
    print(f"Overall Disagreement Rate: {disagreement_results['disagreement_percentage']:.2f}%\n")
    print(f"Number of disagreement samples: {len(disagreement_results['disagreement_samples'])}") # type: ignore
    print(f"Total test samples: {len(disagreement_results['prediction_table'])}\n") # type: ignore
    
    print("Disagreement by Grid Position (Top 10):")
    grid_disagreement = disagreement_results["grid_position_disagreement"].head(10)  # type: ignore
    print(grid_disagreement.to_string(index=False))
    print()
    
    return disagreement_results


def stage_7_save_results(
    metrics_baseline: pd.DataFrame,
    metrics_tuned: pd.DataFrame,
    disagreement_results: Dict,
    output_dir: str = ".",
) -> None:
    """Save analysis results to CSV files."""
    print_section("STAGE 7: Save Results")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    baseline_path = output_path / "metrics_baseline.csv"
    tuned_path = output_path / "metrics_tuned.csv"
    disagreement_path = output_path / "disagreement_by_grid.csv"
    
    metrics_baseline.to_csv(baseline_path, index=False)
    metrics_tuned.to_csv(tuned_path, index=False)
    disagreement_results["grid_position_disagreement"].to_csv(disagreement_path, index=False)  # type: ignore
    
    print(f"Baseline metrics → {baseline_path}")
    print(f"Tuned metrics → {tuned_path}")
    print(f"Disagreement analysis → {disagreement_path}\n")


def print_summary(metrics_baseline: pd.DataFrame, metrics_tuned: pd.DataFrame) -> None:
    """Print final summary comparison."""
    print_section("FINAL SUMMARY")
    
    best_baseline = metrics_baseline.iloc[0]  # type: ignore
    best_tuned = metrics_tuned.iloc[0]  # type: ignore
    
    print(f"Best Baseline Model: {best_baseline['Model']}")
    print(f"  F1-score: {best_baseline['F1-score']:.4f}")
    print(f"  Accuracy: {best_baseline['Accuracy']:.4f}\n")
    
    print(f"Best Tuned Model: {best_tuned['Model']}")
    print(f"  F1-score: {best_tuned['F1-score']:.4f}")
    print(f"  Accuracy: {best_tuned['Accuracy']:.4f}\n")
    
    f1_improvement = ((best_tuned['F1-score'] - best_baseline['F1-score']) / best_baseline['F1-score']) * 100
    print(f"F1-score improvement after tuning: {f1_improvement:+.2f}%\n")


def main(args: Any) -> None:
    """Execute the complete ML pipeline."""
    print_section("F1 DEVIATION PREDICTION PIPELINE")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}\n")
    
    # Stage 1: Load data
    data = stage_1_load_and_prepare(args.data)
    
    # Stage 2: Train baseline models
    artifacts, label_encoder = stage_2_train_baseline_models(data)
    
    # Stage 3: Evaluate baseline models
    metrics_baseline, cm_baseline, preds_baseline = stage_3_evaluate_baseline(artifacts, label_encoder)  # type: ignore
    
    # Stage 4: Hyperparameter tuning
    tuning_results = stage_4_hyperparameter_tuning(artifacts)
    
    # Stage 5: Evaluate tuned models
    metrics_tuned, cm_tuned, preds_tuned = stage_5_evaluate_tuned_models(tuning_results, artifacts, label_encoder)  # type: ignore
    
    # Stage 6: Disagreement analysis
    disagreement_results = stage_6_disagreement_analysis(artifacts, tuning_results, label_encoder)  # type: ignore
    
    # Stage 7: Save results
    stage_7_save_results(metrics_baseline, metrics_tuned, disagreement_results, args.output)  # type: ignore
    
    # Final summary
    print_summary(metrics_baseline, metrics_tuned)  # type: ignore
    
    print_section("PIPELINE COMPLETE")
    print("All analysis results saved to output directory.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end ML pipeline for F1 deviation prediction."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="f1_deviation_dataset_2022_2024.csv",
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Directory to save output files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
