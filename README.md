# Predictive Modeling with Machine Learning and Neural Networks: Wind Turbine Failure Detection

**Live case-study report:** [View the GitHub Pages report](https://namees-albayati.github.io/wind-turbine-failure-detection/)

## Business Context

Aeolus Renewables operates onshore wind turbines under long-term power purchase agreements, making reliable electricity delivery important to revenue. Gearbox and main-bearing failures are operationally significant because they can cause extended outages, emergency repair costs, and additional drivetrain damage. Earlier identification of credible fault conditions could help reduce unplanned downtime and support better maintenance decisions.

The fleet size, installed capacity, typical downtime, and cost context in this case study are assumptions supplied in the project brief. They are not findings calculated from the CSV.

## Problem Statement

This project develops a proof-of-concept binary classifier that predicts `failure` from 10-minute SCADA sensor records collected from 15 wind turbines.

- `0` = normal drivetrain condition
- `1` = drivetrain fault detected

The sole source dataset is `data/raw/wind_turbine_detection.csv`.

## Business Objective

The model is intended to help operations analysts triage credible drivetrain risks and help maintenance planners prioritize inspections and interventions. Detecting genuine failures is more important than maximizing raw accuracy because a false negative represents a missed real fault.

The model is a decision-support tool. It is not an autonomous turbine shutdown system, and its alerts require review by qualified analysts and technicians.

## Evaluation and Model Selection

Failure-class recall is the primary model-selection metric:

`Recall = True Positives / (True Positives + False Negatives)`

Recall is prioritized because false negatives are actual drivetrain faults that the model fails to detect. Accuracy alone will not determine the best model.

Evaluation will also include:

- Precision for the failure class
- F1 score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Classification report

The final classification threshold will be selected using validation data and then frozen before the final test evaluation. The untouched test set will be used once to estimate final model performance.

## Methodology

1. Assess data structure and quality, then apply documented preprocessing.
2. Conduct univariate, bivariate, and multivariate exploratory analysis.
3. Create a leakage-aware, time-based train/validation/test split.
4. Train baseline Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Artificial Neural Network models.
5. Tune the two strongest baseline candidates using validation results, with failure recall as the primary selection criterion.
6. Compare final candidates, select and freeze a decision threshold, and evaluate once on the test set.
7. Analyze feature importance using methods appropriate to the selected model.
8. Translate validated findings into business recommendations and clearly stated limitations.

All empirical observations, preprocessing decisions, model comparisons, and recommendations will be based on executed analysis of the supplied CSV.

## Repository Structure

Current repository structure:

```text
.
├── data/
│   ├── raw/
│   │   └── wind_turbine_detection.csv
│   └── processed/
├── models/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── .gitignore
└── README.md
```

- `data/raw/`: immutable source data supplied for the case study.
- `data/processed/`: cleaned or model-ready data, created only if needed.
- `models/`: trained models and fitted preprocessing pipelines.
- `notebooks/`: numbered notebooks for data understanding, quality checks, EDA, and modeling.
- `reports/figures/`: for model-evaluation visualizations.
- `reports/model_results/`: compact CSV files containing metrics, tuning results, and feature-importance tables; this directory will be created when those outputs are generated.
- `src/`: reusable analysis, preprocessing, training, and evaluation code.

Every notebook must end with a markdown cell titled `## Key Findings Recap`. It must be the last cell and contain five to eight concise factual bullets based only on results executed and reviewed in that notebook.

## Reproducibility

From the project root, create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies after `requirements.txt` is added with the analysis workflow:

```bash
pip install -r requirements.txt
```

Launch Jupyter and open the primary notebook:

```bash
jupyter lab notebooks/01_data_understanding.ipynb
```

Run each notebook from top to bottom in numerical order. Every notebook must execute without errors in a clean environment. Final results must be generated programmatically from `data/raw/wind_turbine_detection.csv`; metrics and findings must not be entered manually.

## Limitations and Responsible Use

- This proof of concept uses historical data from 15 turbines and requires further validation before fleet-wide use.
- Observed model relationships are associations and do not establish causation.
- Model alerts require review by operations analysts and qualified technicians before maintenance action.
- Deployment would require monitoring for sensor and data drift, model-performance degradation, and changing operating conditions.
- Any deployed model would require periodic review and retraining using appropriately validated recent data.

## Expected Outputs

- Cleaned or model-ready data, only if needed: `data/processed/`
- Trained final model and preprocessing pipelines: `models/`
- Exported charts: `reports/figures/`
- Metrics, tuning results, and feature-importance tables: `reports/model_results/`
- Executive report: `reports/Wind_Turbine_Failure_Detection_Report.md`

## Project Status

The business objective and planned methodology are defined. EDA and modeling findings will be added only after the workflow has been executed on the supplied dataset. No dataset results or model-performance claims are reported at this stage.
