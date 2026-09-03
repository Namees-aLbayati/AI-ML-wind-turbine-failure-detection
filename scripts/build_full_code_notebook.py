"""Build the single-file, full-code case-study submission notebook."""

from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
OUTPUT = NOTEBOOK_DIR / "AIML_Project_1_Full_Code_Notebook_Completed.ipynb"

SOURCES = [
    "01_data_understanding.ipynb",
    "02_eda.ipynb",
    "03_preprocessing.ipynb",
    "04_feature_engineering.ipynb",
    "05_baseline_modeling.ipynb",
    "06_hyperparameter_tuning.ipynb",
    "07_final_model_evaluation.ipynb",
]


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    merged = nbf.v4.new_notebook()
    merged.metadata = nbf.read(NOTEBOOK_DIR / SOURCES[0], as_version=4).metadata.copy()
    merged.metadata["title"] = "Wind Turbine Failure Detection — Full-code Version"

    merged.cells.extend(
        [
            markdown(
                """
# Predictive Modeling with Machine Learning and Neural Networks
## Wind Turbine Failure Detection — Completed Full-code Version

This notebook is the consolidated, reproducible submission for the wind-turbine
failure-detection case study. It combines the complete executed workflow from the
numbered project notebooks while preserving the original code, figures, diagnostics,
and reviewed results.

**Business objective:** identify credible drivetrain faults early enough to support
inspection and maintenance prioritization. Failure-class recall is the primary model
selection criterion because a false negative represents a genuine fault that was missed.
Precision, F1, PR-AUC, ROC-AUC, confusion matrices, and classification reports provide
supporting evidence.

**Responsible-use boundary:** this is a decision-support proof of concept, not an
autonomous turbine-shutdown system. Alerts require review by qualified operations and
maintenance personnel.
"""
            ),
            markdown(
                """
## Rubric Coverage and Notebook Guide

| Rubric requirement | Location in this notebook |
|---|---|
| Problem definition and data overview | Part 1 |
| Univariate, bivariate, and multivariate EDA | Part 2 |
| Missing values, feature policy, and chronological split | Part 3 |
| Feature engineering with rationale | Part 4 |
| Five required baseline classifiers | Part 5 |
| Tuning of two selected models | Part 6 |
| Final comparison, selection, test evaluation, and feature importance | Part 7 and Submission Summary |
| Actionable insights and recommendations | Submission Summary |

The original analysis is organized into separate development notebooks in the repository.
This consolidated version is provided for the required full-code HTML/IPYNB submission.
"""
            ),
            markdown(
                """
## Methodological Guardrails

- The 10-minute panel data are split chronologically using shared timestamp boundaries.
- Failure episodes are kept intact across partitions to reduce temporal leakage.
- Imputation, transformations, scaling, and feature selection are learned from training data only.
- Historical fields with uncertain real-time provenance are excluded from model predictors.
- Engineered rolling variables use current and past observations only—never future readings.
- Model and threshold selection use validation data; the test set is reserved for final evaluation.
- The Artificial Neural Network baseline is implemented with scikit-learn's `MLPClassifier`,
  a feed-forward multilayer perceptron classifier. TensorFlow/Keras is not required to satisfy
  the model-family requirement.
"""
            ),
        ]
    )

    for index, filename in enumerate(SOURCES, start=1):
        source = nbf.read(NOTEBOOK_DIR / filename, as_version=4)
        merged.cells.append(markdown(f"# Part {index} — Source: `{filename}`"))
        merged.cells.extend(source.cells)

    summary_cells = [
            markdown(
                """
# Submission Summary

The following compact tables make the model comparisons requested by the full-code
template explicit. They load the executed artifacts produced in Parts 5–7; no metric is
entered manually.
"""
            ),
            code(
                """
from pathlib import Path
import pandas as pd

results_dir = Path("../reports/model_results")
if not results_dir.exists():
    results_dir = Path("reports/model_results")

baseline_submission = pd.read_csv(results_dir / "05_baseline_metrics.csv")
baseline_submission.columns.tolist(), baseline_submission.shape
"""
            ),
            code(
                """
train_columns = [
    column for column in [
        "model", "train_recall", "train_precision", "train_f1",
        "train_pr_auc", "train_roc_auc"
    ] if column in baseline_submission.columns
]
validation_columns = [
    column for column in [
        "model", "validation_recall", "validation_precision", "validation_f1",
        "validation_pr_auc", "validation_roc_auc", "recall_gap"
    ] if column in baseline_submission.columns
]

print("Baseline training performance")
display(baseline_submission[train_columns].round(4))
print("Baseline validation performance")
display(baseline_submission[validation_columns].round(4))
"""
            ),
            code(
                """
tuned_submission = pd.read_csv(results_dir / "06_tuned_validation_metrics.csv")
print("Tuned-model validation comparison")
display(tuned_submission.round(4))

final_submission = pd.read_csv(results_dir / "07_final_test_metrics.csv")
print("Frozen final-model test performance")
display(final_submission.round(4))
"""
            ),
            code(
                """
importance_submission = pd.read_csv(
    results_dir / "07_final_xgboost_importance_interpretation.csv"
)
print("Top original and engineered predictors used by the selected model")
display(importance_submission.head(15).round(4))
"""
            ),
            markdown(
                """
# Final Findings and Recommendations

## Final Findings

- XGBoost was selected after baseline comparison and leakage-aware chronological tuning.
- With the validation-selected threshold frozen at **0.035**, the final test recall was
  **0.8518**, precision **0.6116**, F1 **0.7120**, PR-AUC **0.8548**, and ROC-AUC **0.9859**.
- The final model detected **60 of 65** distinct test failure episodes.
- Drivetrain and tower vibration signals were the most influential sensor families.
- Both current readings and past-only rolling features contributed useful predictive signal;
  importance describes model use and does not establish physical causation.

## Actionable Business Recommendations

1. Use the model as a risk-prioritization layer for analyst review rather than an automatic shutdown rule.
2. Prioritize inspection when drivetrain or tower-vibration alerts persist across consecutive readings.
3. Combine model alerts with technician checks, maintenance history, and operating context before intervention.
4. Pilot the workflow on a limited turbine group and track alert precision, missed episodes, lead time, and maintenance outcomes.
5. Monitor input drift and recall by turbine, site, season, and operating regime; recalibrate the threshold when costs or capacity change.
6. Validate externally on newer periods and additional wind farms before production deployment.

## Limitations

- The historical dataset covers 15 turbines and may not represent other assets or sites.
- Sensor relationships are associative, not causal.
- The low operating threshold favors detection and therefore creates false-positive workload.
- Maintenance cost savings and real-world alert lead time require prospective operational validation.

## Repository

[GitHub repository](https://github.com/Namees-aLbayati/wind-turbine-failure-detection) ·
[Published report](https://namees-albayati.github.io/wind-turbine-failure-detection/)
"""
            ),
        ]

    merged.cells.extend(
        [
            *summary_cells,
        ]
    )

    # Render the submission-only summaries from saved, executed artifacts.
    results_dir = ROOT / "reports" / "model_results"
    baseline = pd.read_csv(results_dir / "05_baseline_metrics.csv")
    tuned = pd.read_csv(results_dir / "06_tuned_validation_metrics.csv")
    final = pd.read_csv(results_dir / "07_final_test_metrics.csv")
    importance = pd.read_csv(results_dir / "07_final_xgboost_importance_interpretation.csv").head(15)
    train_cols = ["model", "train_recall", "train_precision", "train_f1", "train_pr_auc", "train_roc_auc"]
    validation_cols = ["model", "validation_recall", "validation_precision", "validation_f1", "validation_pr_auc", "validation_roc_auc", "recall_gap"]

    summary_cells[1].outputs = [nbf.v4.new_output("execute_result", data={"text/plain": repr((baseline.columns.tolist(), baseline.shape))}, execution_count=None)]
    summary_cells[2].outputs = [
        nbf.v4.new_output("stream", name="stdout", text="Baseline training performance\n"),
        nbf.v4.new_output("display_data", data={"text/plain": baseline[train_cols].round(4).to_string(index=False), "text/html": baseline[train_cols].round(4).to_html(index=False)}),
        nbf.v4.new_output("stream", name="stdout", text="Baseline validation performance\n"),
        nbf.v4.new_output("display_data", data={"text/plain": baseline[validation_cols].round(4).to_string(index=False), "text/html": baseline[validation_cols].round(4).to_html(index=False)}),
    ]
    summary_cells[3].outputs = [
        nbf.v4.new_output("stream", name="stdout", text="Tuned-model validation comparison\n"),
        nbf.v4.new_output("display_data", data={"text/plain": tuned.round(4).to_string(index=False), "text/html": tuned.round(4).to_html(index=False)}),
        nbf.v4.new_output("stream", name="stdout", text="Frozen final-model test performance\n"),
        nbf.v4.new_output("display_data", data={"text/plain": final.round(4).to_string(index=False), "text/html": final.round(4).to_html(index=False)}),
    ]
    summary_cells[4].outputs = [
        nbf.v4.new_output("stream", name="stdout", text="Top original and engineered predictors used by the selected model\n"),
        nbf.v4.new_output("display_data", data={"text/plain": importance.round(4).to_string(index=False), "text/html": importance.round(4).to_html(index=False)}),
    ]

    # A merged notebook retains executed evidence but receives unique IDs and a clean display order.
    execution_number = 1
    for index, cell in enumerate(merged.cells):
        cell.id = f"full-code-{index:03d}"
        if cell.cell_type == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                cell.execution_count = execution_number
                for output in cell.get("outputs", []):
                    if output.output_type == "execute_result":
                        output.execution_count = execution_number
                execution_number += 1

    nbf.write(merged, OUTPUT)
    print(f"Created {OUTPUT.relative_to(ROOT)} with {len(merged.cells)} cells")


if __name__ == "__main__":
    main()
