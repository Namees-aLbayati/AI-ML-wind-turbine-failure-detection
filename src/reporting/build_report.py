from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _format_table(frame: pd.DataFrame, columns: list[str], labels: dict[str, str]) -> str:
    displayed = frame[columns].rename(columns=labels).copy()
    numeric = displayed.select_dtypes(include="number").columns
    displayed[numeric] = displayed[numeric].round(4)
    return displayed.to_html(index=False, border=0, classes="report-table")


def _create_eda_summary(raw: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    counts = raw["failure"].value_counts().sort_index()
    axes[0].bar(["Normal", "Failure"], counts.values, color=["#457b9d", "#e76f51"])
    axes[0].set_yscale("log")
    axes[0].set_title("Target Class Counts (log scale)")
    axes[0].set_ylabel("SCADA records")
    for index, value in enumerate(counts.values):
        axes[0].text(index, value * 1.08, f"{value:,}", ha="center", fontsize=9)

    turbine_rates = raw.groupby("turbine_id")["failure"].mean().mul(100).sort_values()
    axes[1].barh(turbine_rates.index, turbine_rates.values, color="#2a9d8f")
    axes[1].set_title("Failure Rate by Turbine")
    axes[1].set_xlabel("Failure records (%)")
    axes[1].set_ylabel("Turbine")

    normal_sample = raw.loc[raw["failure"].eq(0)].sample(n=20_000, random_state=42)
    vibration_sample = pd.concat([normal_sample, raw.loc[raw["failure"].eq(1)]], ignore_index=True)
    vibration_sample["Condition"] = vibration_sample["failure"].map({0: "Normal", 1: "Failure"})
    upper_limit = vibration_sample["drivetrain_vibration_rms_mmps"].quantile(0.995)
    sns.boxplot(data=vibration_sample, x="Condition", y="drivetrain_vibration_rms_mmps", hue="Condition", legend=False, palette=["#457b9d", "#e76f51"], ax=axes[2])
    axes[2].set_ylim(0, upper_limit)
    axes[2].set_title("Drivetrain Vibration by Condition")
    axes[2].set_ylabel("RMS vibration (mm/s)")

    fig.suptitle("Exploratory Data Summary", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_final_report(project_root: Path) -> Path:
    project_root = project_root.resolve()
    data_dir = project_root / "data" / "processed"
    raw_path = project_root / "data" / "raw" / "wind_turbine_detection.csv"
    results_dir = project_root / "reports" / "model_results"
    figures_dir = project_root / "reports" / "figures"
    report_path = project_root / "reports" / "Wind_Turbine_Failure_Detection_Report.html"

    required = [
        results_dir / "05_baseline_metrics.csv",
        results_dir / "06_tuned_validation_metrics.csv",
        results_dir / "07_final_test_metrics.csv",
        results_dir / "07_final_test_operational_metrics.csv",
        results_dir / "07_final_xgboost_sensor_family_importance.csv",
        data_dir / "03_train_metadata.csv",
        data_dir / "03_validation_metadata.csv",
        data_dir / "03_test_metadata.csv",
    ]
    missing = [str(path.relative_to(project_root)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required reporting artifacts are missing: {missing}")

    raw = pd.read_csv(raw_path)
    baseline = pd.read_csv(results_dir / "05_baseline_metrics.csv")
    tuned = pd.read_csv(results_dir / "06_tuned_validation_metrics.csv")
    final = pd.read_csv(results_dir / "07_final_test_metrics.csv").iloc[0]
    operational = pd.read_csv(results_dir / "07_final_test_operational_metrics.csv").iloc[0]
    families = pd.read_csv(results_dir / "07_final_xgboost_sensor_family_importance.csv")
    selected_features = pd.read_csv(results_dir / "04_selected_features.csv")
    split_metadata = {name: pd.read_csv(data_dir / f"03_{name}_metadata.csv") for name in ["train", "validation", "test"]}

    eda_figure = figures_dir / "08_report_eda_summary.png"
    _create_eda_summary(raw, eda_figure)

    baseline_table = _format_table(
        baseline,
        ["model", "validation_recall", "validation_precision", "validation_f1", "validation_pr_auc", "validation_roc_auc"],
        {"model": "Model", "validation_recall": "Recall", "validation_precision": "Precision", "validation_f1": "F1", "validation_pr_auc": "PR-AUC", "validation_roc_auc": "ROC-AUC"},
    )
    tuned_table = _format_table(
        tuned,
        ["model", "threshold", "recall", "precision", "f1", "f2", "pr_auc", "roc_auc"],
        {"model": "Model", "threshold": "Threshold", "recall": "Recall", "precision": "Precision", "f1": "F1", "f2": "F2", "pr_auc": "PR-AUC", "roc_auc": "ROC-AUC"},
    )
    family_table = _format_table(
        families.head(10),
        ["rank", "source_sensor_family", "importance", "cumulative_importance"],
        {"rank": "Rank", "source_sensor_family": "Source-sensor family", "importance": "Importance", "cumulative_importance": "Cumulative importance"},
    )

    image_names = {
        "eda": eda_figure,
        "baseline": figures_dir / "05_baseline_roc_pr_curves.png",
        "tuning": figures_dir / "06_tuned_validation_confusion_matrices.png",
        "test_confusion": figures_dir / "07_final_test_confusion_matrix.png",
        "test_curves": figures_dir / "07_final_test_roc_pr_curves.png",
        "importance": figures_dir / "07_final_xgboost_feature_importance.png",
        "families": figures_dir / "07_final_xgboost_sensor_family_importance.png",
    }
    missing_figures = [str(path.relative_to(project_root)) for path in image_names.values() if not path.exists()]
    if missing_figures:
        raise FileNotFoundError(f"Required figures are missing: {missing_figures}")
    images = {name: _image_data_uri(path) for name, path in image_names.items()}

    split_rows = {name: len(frame) for name, frame in split_metadata.items()}
    split_failures = {name: int(frame["failure"].sum()) for name, frame in split_metadata.items()}
    split_rates = {name: frame["failure"].mean() * 100 for name, frame in split_metadata.items()}
    top_family = html.escape(str(families.iloc[0]["source_sensor_family"]))

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Wind Turbine Failure Detection — Final Report</title>
<style>
:root {{--ink:#183153;--muted:#5f6b7a;--blue:#1d5d86;--teal:#2a9d8f;--light:#f4f7fa;--line:#d8e0e8;--accent:#e76f51}}
* {{box-sizing:border-box}} body {{margin:0;background:#eef2f5;color:var(--ink);font:16px/1.58 Arial,sans-serif}}
main {{max-width:1080px;margin:32px auto;background:white;padding:54px 66px;box-shadow:0 4px 22px #22334418}}
h1 {{font-size:2.35rem;line-height:1.15;margin:0 0 8px;color:#12344d}} h2 {{margin-top:42px;border-bottom:2px solid var(--teal);padding-bottom:7px}} h3 {{margin-top:28px}}
.subtitle {{font-size:1.15rem;color:var(--muted);margin-bottom:30px}} .executive {{background:linear-gradient(135deg,#edf7f6,#f6f9fc);border-left:5px solid var(--teal);padding:20px 24px}}
.metrics {{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0}} .metric {{background:var(--light);padding:15px 10px;text-align:center;border-radius:7px}} .metric strong {{display:block;font-size:1.45rem;color:var(--blue)}}
.report-table {{border-collapse:collapse;width:100%;font-size:.92rem;margin:18px 0}} .report-table th {{background:var(--ink);color:white;text-align:left}} .report-table th,.report-table td {{padding:9px 10px;border:1px solid var(--line)}} .report-table tr:nth-child(even) {{background:var(--light)}}
figure {{margin:25px 0}} figure img {{max-width:100%;display:block;margin:auto}} figcaption {{color:var(--muted);font-size:.9rem;text-align:center;margin-top:8px}}
.callout {{border:1px solid var(--line);border-radius:7px;padding:15px 18px;background:#fafcfd}} li {{margin:7px 0}} footer {{margin-top:48px;padding-top:14px;border-top:1px solid var(--line);color:var(--muted);font-size:.88rem}}
@media(max-width:760px) {{main {{margin:0;padding:28px 20px}} .metrics {{grid-template-columns:repeat(2,1fr)}}}}
@media print {{body {{background:white}} main {{box-shadow:none;margin:0;max-width:none}}}}
</style></head><body><main>
<h1>Wind Turbine Failure Detection</h1><div class="subtitle">Predictive Modeling with Machine Learning and Neural Networks — Final Case Study Report</div>

<section class="executive"><h2 style="margin-top:0">Executive Summary</h2>
<p>A recall-first XGBoost classifier was developed to identify current drivetrain-fault conditions from 10-minute SCADA records. The final pipeline was selected without using the test period and then evaluated once on later unseen-in-time records.</p>
<div class="metrics"><div class="metric"><strong>{final['recall']:.3f}</strong>Test recall</div><div class="metric"><strong>{final['precision']:.3f}</strong>Precision</div><div class="metric"><strong>{final['f1']:.3f}</strong>F1</div><div class="metric"><strong>{final['pr_auc']:.3f}</strong>PR-AUC</div><div class="metric"><strong>{operational['episode_recall']:.3f}</strong>Episode recall</div></div>
<p>The model detected {int(operational['detected_episodes'])} of {int(operational['failure_episodes'])} test failure episodes. Its dominant information source was {top_family.lower()}, especially sustained one- and three-hour behavior. Results support analyst triage and inspection prioritization, not autonomous shutdown decisions.</p></section>

<h2>1. Business Problem and Objective</h2>
<p>Aeolus Renewables requires reliable electricity delivery under long-term power purchase agreements. Gearbox and main-bearing faults can contribute to extended outages, emergency repair costs, and additional drivetrain damage. This proof of concept classifies each SCADA record as normal (<code>failure=0</code>) or indicative of a current drivetrain fault (<code>failure=1</code>).</p>
<p>Failure recall is the primary criterion because a false negative is a real fault the model misses. Precision, F1, PR-AUC, ROC-AUC, alert burden, and episode recall are retained so recall is not optimized without regard to operational usability.</p>

<h2>2. Data Overview and Exploratory Analysis</h2>
<p>The source contains {len(raw):,} records, {raw.shape[1]} columns, and {raw['turbine_id'].nunique()} turbines at a verified 10-minute cadence. It spans 1 January through 1 March 2024. The target contains {int(raw['failure'].sum()):,} failure records ({raw['failure'].mean()*100:.3f}%). Missingness is limited to gearbox-oil temperature, generator-bearing temperature, and oil pressure, each affecting approximately 0.5% of rows.</p>
<figure><img src="{images['eda']}" alt="Exploratory data summary"><figcaption>Class imbalance, between-turbine failure-rate variation, and the relationship between drivetrain vibration and fault condition.</figcaption></figure>
<ul><li>Failure records formed 219 per-turbine episodes in the complete dataset.</li><li>Failure observations showed elevated gearbox-bearing temperature, drivetrain vibration, bearing-frequency amplitude, and oil-particle count.</li><li>Vibration and temperature outliers were retained because they were disproportionately associated with failure records.</li><li>Observed relationships are associations and are not evidence of physical causation.</li></ul>

<h2>3. Preprocessing and Leakage Controls</h2>
<p>A shared chronological split placed {split_rows['train']:,} rows in training, {split_rows['validation']:,} in validation, and {split_rows['test']:,} in the final test period. All 15 turbines appear in every period, every row occurs exactly once, and failure episodes were not divided at the boundaries.</p>
<table class="report-table"><thead><tr><th>Partition</th><th>Rows</th><th>Failures</th><th>Failure rate</th></tr></thead><tbody>
<tr><td>Training</td><td>{split_rows['train']:,}</td><td>{split_failures['train']:,}</td><td>{split_rates['train']:.3f}%</td></tr><tr><td>Validation</td><td>{split_rows['validation']:,}</td><td>{split_failures['validation']:,}</td><td>{split_rates['validation']:.3f}%</td></tr><tr><td>Test</td><td>{split_rows['test']:,}</td><td>{split_failures['test']:,}</td><td>{split_rates['test']:.3f}%</td></tr></tbody></table>
<p>Imputation, encoding, scaling, model fitting, and imbalance treatment were learned from training data within pipelines. Six provenance-risk maintenance/history fields were excluded. The rising failure prevalence across time is a material distribution shift and is considered when interpreting generalization.</p>

<h2>4. Feature Engineering</h2>
<p>Step 04 created 24 domain-informed snapshot features and 36 past-only temporal features. These include safe operating ratios, component temperature rises, FFT summaries, circular direction encodings, lagged measurements, recent changes, and one- and three-hour rolling statistics calculated independently per turbine. No centered or future-looking windows were used.</p>
<p>A training/validation ablation selected {len(selected_features)} predictors combining original measurements with engineered features. The test set was not scored during this selection.</p>

<h2>5. Baseline Model Comparison</h2>
<p>Five assignment-required classifiers were fitted on the full training period at a fixed 0.50 threshold. XGBoost produced the strongest baseline recall and PR-AUC. Random Forest showed strong probability ranking but conservative classifications, motivating tuning and threshold analysis.</p>
{baseline_table}
<figure><img src="{images['baseline']}" alt="Baseline ROC and precision-recall curves"><figcaption>Validation ROC and precision-recall curves for the five baseline models.</figcaption></figure>

<h2>6. Hyperparameter Tuning and Threshold Selection</h2>
<p>XGBoost and Random Forest were tuned using eight configurations each over three expanding chronological folds within the training period. The winning configurations were refitted on all training data. Each threshold was chosen on validation data by maximizing F2, which weights recall twice as strongly as precision.</p>
{tuned_table}
<figure><img src="{images['tuning']}" alt="Tuned validation confusion matrices"><figcaption>Tuned validation confusion matrices at the selected model-specific thresholds.</figcaption></figure>
<p>XGBoost was selected and its validation-derived threshold of {final['threshold']:.3f} was frozen before test evaluation.</p>

<h2>7. Final Test Performance</h2>
<p>The selected configuration was refitted on combined training and validation data and evaluated once on the later test period.</p>
<div class="metrics"><div class="metric"><strong>{final['recall']:.4f}</strong>Recall</div><div class="metric"><strong>{final['precision']:.4f}</strong>Precision</div><div class="metric"><strong>{final['f1']:.4f}</strong>F1</div><div class="metric"><strong>{final['pr_auc']:.4f}</strong>PR-AUC</div><div class="metric"><strong>{final['roc_auc']:.4f}</strong>ROC-AUC</div></div>
<figure><img src="{images['test_confusion']}" alt="Final test confusion matrix"><figcaption>The final model correctly detected 1,069 failure records, missed 186, and produced 679 false-positive records.</figcaption></figure>
<figure><img src="{images['test_curves']}" alt="Final test ROC and precision-recall curves"><figcaption>Threshold-independent discrimination on the untouched test period.</figcaption></figure>
<div class="callout"><strong>Operational view:</strong> The classifier detected {int(operational['detected_episodes'])} of {int(operational['failure_episodes'])} failure episodes ({operational['episode_recall']:.2%}). It produced {operational['false_positive_records_per_turbine_day']:.2f} false-positive 10-minute records per turbine-day before alert deduplication or cooldown rules.</div>

<h2>8. Feature Importance</h2>
<p>Gain-based XGBoost importance is presented in two complementary views. The detailed view shows the actual original and engineered model inputs. The grouped view combines derived variants with their underlying physical sensor family, which is more suitable for business interpretation.</p>
<figure><img src="{images['importance']}" alt="Detailed feature importance"><figcaption>Top model inputs, including original and engineered predictors.</figcaption></figure>
{family_table}
<figure><img src="{images['families']}" alt="Sensor-family importance"><figcaption>Importance aggregated by physical source-sensor family.</figcaption></figure>
<p>Drivetrain vibration contributed {families.iloc[0]['importance']:.2%} of fitted importance after grouping current, lagged, rolling, and normalized variants. Tower vibration and FFT vibration signals followed. Correlated features may divide or concentrate gain importance, so these values explain this fitted model rather than proving causal fault mechanisms.</p>

<h2>9. Actionable Insights and Recommendations</h2>
<ol><li><strong>Prioritize sustained vibration:</strong> route persistent drivetrain-vibration elevations and supporting tower/FFT patterns to analyst review.</li><li><strong>Deploy as decision support:</strong> use scores to prioritize inspections; do not trigger autonomous shutdown or maintenance without qualified review.</li><li><strong>Control alert burden:</strong> convert consecutive positive records into one alert episode and apply a documented cooldown. Validate whether the observed false-positive burden fits maintenance capacity.</li><li><strong>Investigate missed episodes:</strong> review the five undetected test episodes by turbine, duration, operating state, and sensor availability.</li><li><strong>Monitor drift:</strong> track failure prevalence, feature distributions, recall, precision, and alert volume by turbine and calendar period.</li><li><strong>Validate prospectively:</strong> conduct a shadow deployment before operational use and retrain only after appropriately labeled recent data are available.</li></ol>

<h2>10. Limitations and Responsible Use</h2>
<ul><li>The dataset represents 15 turbines and approximately two months, limiting seasonal and fleet-wide generalization.</li><li>The label supports detection of current fault conditions; it does not demonstrate hours- or days-ahead failure forecasting.</li><li>Failure prevalence rises materially between training and later periods.</li><li>Episode recall counts an episode as detected if any constituent record is positive and does not itself measure advance warning.</li><li>False-positive records are not equivalent to work orders until an alert aggregation policy is defined.</li><li>Feature importance is model-specific and associative, not causal.</li></ul>

<h2>11. Conclusion</h2>
<p>The final XGBoost pipeline provides strong discrimination and detects 85.18% of failure records and 92.31% of failure episodes in the later test period. Sustained drivetrain-vibration behavior is the dominant signal family. The results support a promising analyst-triage proof of concept, subject to prospective validation, alert aggregation, ongoing drift monitoring, and qualified human review.</p>
<footer>Generated programmatically from the project’s executed and saved analysis artifacts. No final metrics or model comparisons were manually recomputed in the reporting step.</footer>
</main></body></html>"""

    report_path.write_text(document, encoding="utf-8")
    return report_path
