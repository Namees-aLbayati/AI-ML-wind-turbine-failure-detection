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

Failure-class recall expresses the detection objective; final pipeline selection uses validation PR-AUC, and the decision threshold maximizes validation F2:

`Recall = True Positives / (True Positives + False Negatives)`

Recall is prioritized because false negatives are actual drivetrain faults that the model fails to detect. Accuracy alone will not determine the best model.

Evaluation will also include:

- Precision for the failure class
- F1 score
- ROC-AUC
- PR-AUC
- Confusion matrix
- Classification report

The final classification threshold will be selected using validation data and then frozen before the final test evaluation. The held-out-turbine test set estimates final performance. Earlier exploratory test exposure is disclosed; it is not fresh independent confirmation.

## Methodology

1. Assess data structure and quality, then apply documented preprocessing.
2. Conduct univariate, bivariate, and multivariate exploratory analysis.
3. Hold out complete turbines for testing and use later development observations for validation.
4. Train baseline Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Artificial Neural Network models.
5. Tune the prespecified XGBoost and Random Forest candidates, then compare them with the selected Keras pipeline using validation PR-AUC, with F2 and recall as tie-breakers.
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

Install the complete project dependencies, including TensorFlow for the Keras experiments:

```bash
pip install -r requirements-neural.txt
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

## Current workflow and execution order

The corrected grouped-split workflow includes five Keras experiments and selects the final
pipeline from validation results. Install `requirements-neural.txt` and run:

```bash
python scripts/run_notebooks.py 03_preprocessing.ipynb 04_feature_engineering.ipynb 05_baseline_modeling.ipynb 05b_neural_networks.ipynb 06_hyperparameter_tuning.ipynb 07_final_model_evaluation.ipynb 08_final_reporting.ipynb
```

Notebooks 01 and 02 contain the initial data audit and EDA. Stage 03 exports the exact
split metadata, fitted ANN preprocessor, arrays and integrity manifest. Stage 04 builds
engineered inputs; stage 05 trains the five traditional baselines. Stage 05b compares
five Keras architectures on the same stage-03 inputs and stops after validation.
Stage 06 tunes two tree candidates and compares complete pipelines with the Keras
candidate using identical validation source rows. Stage 07 evaluates the frozen
checkpoint without refitting and produces model-family-independent artifacts.
Stage 08 verifies fingerprints and rebuilds the standalone report and local site files.

Tree pipelines use selected engineered inputs; Keras uses stage-03 transformed snapshot
inputs. Cross-family results compare complete pipelines, not architecture alone. The
old chronological-split results are superseded. Neither test outcomes nor the identity
of a previously selected model determine the new selection. Earlier exploratory test
exposure and repeated validation use are disclosed in the report.

Generated model files live under `models/` and are git-ignored. The full-code notebook
and HTML include stage 05b and shared implementation sources. Local site generation
does not publish or deploy to a remote service.

## Template-aligned final submission

The primary submission is `notebooks/AIML_Project_1_Full_Code_Notebook_Completed.ipynb`,
with its HTML export at `reports/AIML_Project_1_Full_Code_Notebook_Completed.html`.
It follows the uploaded assignment's analysis headings and order, omitting student instructions and task prompts; the reference is preserved
under `notebooks/reference/`. The previous full-code notebook is retained under
`notebooks/archive/` along with the unchanged development notebooks 01–08.

The template-aligned run reuses the established grouped partitions and selected
engineered inputs. It evaluates all five required baseline families, selects two tree
families by baseline validation recall (then PR-AUC), and evaluates a regularized ANN.
Final selection compares all baseline and tuned candidates by validation recall, then
PR-AUC and precision, at a fixed 0.50 threshold. This is a separate, template-aligned
analysis; its results in `models/template_aligned/` supersede the old metrics only for
the revised notebook. The existing standalone executive report/site still describes
the earlier PR-AUC/F2 workflow above. Neither workflow is fresh external validation.

After the processed inputs have been generated by stages 03 and 04:

```bash
python scripts/align_template_notebook.py
```

To validate the recorded execution and regenerate only its HTML:

```bash
python scripts/build_full_code_notebook.py
```
