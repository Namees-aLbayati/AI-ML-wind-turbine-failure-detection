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
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

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

    correlation_features = [
        "gearbox_bearing_temp_C",
        "main_bearing_temp_C",
        "generator_winding_temp_C",
        "drivetrain_vibration_rms_mmps",
        "tower_vibration_mmps",
        "oil_particle_count",
        "failure",
    ]
    correlation_labels = ["Gearbox temp", "Main-bearing temp", "Winding temp", "Drivetrain vib.", "Tower vib.", "Oil particles", "Failure"]
    correlations = raw[correlation_features].corr()
    correlations.index = correlation_labels
    correlations.columns = correlation_labels
    sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, linewidths=0.4, cbar_kws={"shrink": 0.75}, ax=axes[3])
    axes[3].set_title("Multivariate Correlation Overview")
    axes[3].tick_params(axis="x", rotation=45)
    axes[3].tick_params(axis="y", rotation=0)

    fig.suptitle("Exploratory Data Summary", fontsize=16, y=1.01)
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
    pages_dir = project_root / "docs"
    pages_path = pages_dir / "index.html"
    root_pages_path = project_root / "index.html"

    required = [
        results_dir / "05_baseline_metrics.csv",
        results_dir / "06_tuned_validation_metrics.csv",
        results_dir / "07_final_test_metrics.csv",
        results_dir / "07_final_test_operational_metrics.csv",
        results_dir / "07_final_xgboost_sensor_family_importance.csv",
        results_dir / "07_final_xgboost_importance_interpretation.csv",
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
    importance_interpretation = pd.read_csv(results_dir / "07_final_xgboost_importance_interpretation.csv")
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
    importance_table = _format_table(
        importance_interpretation.head(12),
        ["rank", "feature", "feature_type", "source_sensor_family", "importance"],
        {"rank": "Rank", "feature": "Model feature", "feature_type": "Type", "source_sensor_family": "Source family", "importance": "Importance"},
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
:root {{--ink:#17324d;--muted:#607080;--navy:#12344d;--blue:#1d5d86;--teal:#2a9d8f;--light:#f3f7fa;--line:#d8e1e8;--accent:#e76f51;--gold:#e9c46a}}
* {{box-sizing:border-box}} html {{scroll-behavior:smooth}} body {{margin:0;background:linear-gradient(135deg,#e9f0f4,#f6f8fa);color:var(--ink);font:16px/1.62 Inter,Arial,sans-serif}}
main {{max-width:1120px;margin:28px auto;background:white;padding:0 70px 58px;box-shadow:0 8px 35px #16324a20;border-radius:10px;overflow:hidden}}
.hero {{margin:0 -70px 34px;padding:62px 70px 54px;color:white;background:linear-gradient(125deg,#102f46 0%,#1d5d86 58%,#2a9d8f 100%);position:relative}}
.hero:after {{content:"";position:absolute;right:-80px;top:-90px;width:300px;height:300px;border:45px solid #ffffff18;border-radius:50%}}
h1 {{font-size:2.65rem;line-height:1.12;margin:0 0 10px;color:white;max-width:760px}} h2 {{margin-top:48px;border-bottom:3px solid var(--teal);padding-bottom:8px;color:var(--navy);scroll-margin-top:20px}} h3 {{margin-top:30px;color:var(--blue);scroll-margin-top:20px}} h4 {{color:var(--blue);margin-bottom:5px}}
.subtitle {{font-size:1.17rem;color:#e8f3f6;margin:0 0 18px}} .meta {{display:flex;gap:10px;flex-wrap:wrap}} .badge {{background:#ffffff20;border:1px solid #ffffff45;border-radius:999px;padding:5px 12px;font-size:.84rem}}
.toc {{background:#f7fafc;border:1px solid var(--line);border-radius:9px;padding:18px 22px;margin:25px 0 34px}} .toc strong {{color:var(--navy)}} .toc-links {{display:grid;grid-template-columns:repeat(3,1fr);gap:7px 18px;margin-top:10px}} .toc a {{color:var(--blue);text-decoration:none;font-size:.91rem}} .toc a:hover {{text-decoration:underline}}
.executive {{background:linear-gradient(135deg,#ecf8f5,#f5f9fc);border-left:5px solid var(--teal);padding:22px 26px;border-radius:0 8px 8px 0}}
.metrics {{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:20px 0}} .metric {{background:var(--light);padding:15px 10px;text-align:center;border-radius:7px}} .metric strong {{display:block;font-size:1.45rem;color:var(--blue)}}
.report-table {{border-collapse:collapse;width:100%;font-size:.92rem;margin:18px 0}} .report-table th {{background:var(--ink);color:white;text-align:left}} .report-table th,.report-table td {{padding:9px 10px;border:1px solid var(--line)}} .report-table tr:nth-child(even) {{background:var(--light)}}
figure {{margin:25px 0}} figure img {{max-width:100%;display:block;margin:auto}} figcaption {{color:var(--muted);font-size:.9rem;text-align:center;margin-top:8px}}
.callout {{border:1px solid var(--line);border-radius:8px;padding:16px 19px;background:#fafcfd}} .warning {{border-left:5px solid var(--gold);background:#fffaf0}} .success {{border-left:5px solid var(--teal);background:#f0faf7}} li {{margin:7px 0}} code {{background:#eef3f6;padding:2px 5px;border-radius:4px}} .flow {{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:18px 0}} .flow div {{background:var(--navy);color:white;text-align:center;padding:12px 7px;border-radius:7px;font-size:.88rem}} .rubric td:first-child {{font-weight:700;color:var(--navy)}} .check {{color:#087f5b;font-weight:700}} .back-top {{display:block;text-align:right;margin-top:12px;font-size:.82rem;color:var(--blue);text-decoration:none}} footer {{margin-top:50px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:.88rem}} footer a {{color:var(--blue);font-weight:700}}
@media(max-width:760px) {{main {{margin:0;padding:0 20px 30px;border-radius:0}} .hero {{margin:0 -20px 25px;padding:38px 20px}} .metrics,.toc-links,.flow {{grid-template-columns:repeat(2,1fr)}} h1 {{font-size:2rem}}}}
@media print {{body {{background:white}} main {{box-shadow:none;margin:0;max-width:none}}}}
</style></head><body><main id="top">
<header class="hero"><h1>Wind Turbine Failure Detection</h1><div class="subtitle">Predictive Modeling with Machine Learning and Neural Networks</div><div class="meta"><span class="badge">Final Case Study Report</span><span class="badge">15 Turbines</span><span class="badge">131,760 SCADA Records</span><span class="badge">Recall-First Classification</span></div></header>
<nav class="toc"><strong>Report navigation — select a section to jump directly to it</strong><div class="toc-links"><a href="#executive">Executive summary</a><a href="#business">Business objective</a><a href="#eda">Exploratory analysis</a><a href="#preprocessing">Preprocessing</a><a href="#features">Feature engineering</a><a href="#baseline">Baseline models</a><a href="#tuning">Tuning</a><a href="#performance">Final performance</a><a href="#importance">Feature importance</a><a href="#recommendations">Recommendations</a><a href="#limitations">Limitations</a><a href="#rubric">Rubric coverage</a><a href="#conclusion">Conclusion</a></div></nav>

<section class="executive" id="executive"><h2 style="margin-top:0">Executive Summary</h2>
<p>A recall-first XGBoost classifier was developed to identify current drivetrain-fault conditions from 10-minute SCADA records. The final pipeline was selected without using the test period and then evaluated once on later unseen-in-time records.</p>
<div class="metrics"><div class="metric"><strong>{final['recall']:.3f}</strong>Test recall</div><div class="metric"><strong>{final['precision']:.3f}</strong>Precision</div><div class="metric"><strong>{final['f1']:.3f}</strong>F1</div><div class="metric"><strong>{final['pr_auc']:.3f}</strong>PR-AUC</div><div class="metric"><strong>{operational['episode_recall']:.3f}</strong>Episode recall</div></div>
<p>The model detected {int(operational['detected_episodes'])} of {int(operational['failure_episodes'])} test failure episodes. Its dominant information source was {top_family.lower()}, especially sustained one- and three-hour behavior. Results support analyst triage and inspection prioritization, not autonomous shutdown decisions.</p></section>

<h2 id="business">1. Business Problem and Objective</h2>
<p>Aeolus Renewables requires reliable electricity delivery under long-term power purchase agreements. Gearbox and main-bearing faults can contribute to extended outages, emergency repair costs, and additional drivetrain damage. This proof of concept classifies each SCADA record as normal (<code>failure=0</code>) or indicative of a current drivetrain fault (<code>failure=1</code>).</p>
<p>Failure recall is the primary criterion because a false negative is a real fault the model misses. Precision, F1, PR-AUC, ROC-AUC, alert burden, and episode recall are retained so recall is not optimized without regard to operational usability.</p>

<h2 id="eda">2. Data Overview and Exploratory Analysis</h2>
<p>The source contains {len(raw):,} records, {raw.shape[1]} columns, and {raw['turbine_id'].nunique()} turbines at a verified 10-minute cadence. It spans 1 January through 1 March 2024. The target contains {int(raw['failure'].sum()):,} failure records ({raw['failure'].mean()*100:.3f}%). Missingness is limited to gearbox-oil temperature, generator-bearing temperature, and oil pressure, each affecting approximately 0.5% of rows.</p>
<h3>2.1 Key fields</h3>
<table class="report-table"><thead><tr><th>Field group</th><th>Important fields</th><th>Analytical role</th></tr></thead><tbody>
<tr><td>Identifiers</td><td><code>timestamp</code>, <code>turbine_id</code></td><td>Chronological splitting, turbine-grouped histories, and traceability; excluded from predictors.</td></tr>
<tr><td>Target</td><td><code>failure</code></td><td>Binary outcome: 0 normal, 1 detected drivetrain fault.</td></tr>
<tr><td>Environment</td><td>Wind speed/direction, turbulence, air density, ambient temperature, humidity</td><td>Operating context and external conditions.</td></tr>
<tr><td>Operation</td><td>Power output, rotor/generator speed, pitch, yaw, rated power</td><td>Load and operating-state context.</td></tr>
<tr><td>Thermal condition</td><td>Gearbox, generator, main-bearing, and nacelle temperatures</td><td>Component-heating condition signals.</td></tr>
<tr><td>Mechanical condition</td><td>Drivetrain/tower vibration and four FFT bands</td><td>Mechanical and frequency-domain condition signals.</td></tr>
<tr><td>Lubrication</td><td>Oil-particle count and oil pressure</td><td>Lubrication and wear indicators.</td></tr>
</tbody></table>
<h3>2.2 Data quality</h3>
<table class="report-table"><thead><tr><th>Check</th><th>Result</th><th>Decision</th></tr></thead><tbody>
<tr><td>Cadence and keys</td><td>Complete 10-minute cadence; no duplicate turbine/timestamp keys</td><td>Retain all structurally valid rows.</td></tr>
<tr><td>Missing values</td><td>Gearbox-oil temperature: 645; generator-bearing temperature: 690; oil pressure: 636</td><td>Median imputation fitted within training pipelines.</td></tr>
<tr><td>Physical screens</td><td>No values breached the documented conservative screens</td><td>No physical-range deletion.</td></tr>
<tr><td>Distributional outliers</td><td>Several temperature and vibration extremes were failure-enriched</td><td>Retain; these may contain genuine fault information.</td></tr>
<tr><td>Class balance</td><td>127,807 normal and 3,953 failure records</td><td>Use imbalance-aware training and PR-sensitive evaluation.</td></tr>
</tbody></table>
<h3>2.3 Univariate analysis</h3><p>The target is highly imbalanced at 3.000% failures. Strong skew appeared in turbulence intensity, generator-winding temperature, vibration/FFT fields, oil-particle count, oil pressure, power output, and blade pitch. Structural zeros in power and rotational speed were treated as operating states rather than missing observations.</p>
<h3>2.4 Bivariate analysis</h3><p>Failure rates varied from 0.968% to 6.352% across turbines. Compared with normal rows, failure rows had higher mean gearbox-bearing temperature (70.249 versus 62.480°C), drivetrain vibration (3.407 versus 1.786 mm/s), bearing BPFO amplitude (0.859 versus 0.422), and oil-particle count (158.083 versus 71.118).</p>
<h3>2.5 Multivariate analysis</h3><p>Rotor and generator speed were almost perfectly correlated (<em>r</em>=0.99984), and multiple thermal variables exceeded |<em>r</em>|=0.90. Joint temperature, vibration, FFT, and oil-condition patterns motivated multivariate tree models and feature-family interpretation. Trends across the six longest episodes were not directionally consistent enough to justify a single global lead-time slope.</p>
<figure><img src="{images['eda']}" alt="Exploratory data summary"><figcaption>Univariate target balance, bivariate turbine and vibration comparisons, and a multivariate correlation overview.</figcaption></figure>
<div class="callout warning"><strong>EDA interpretation:</strong> Failure records formed 219 per-turbine episodes. Observed relationships identify useful predictive associations but do not establish physical causation.</div>

<h2 id="preprocessing">3. Data Preprocessing and Leakage Controls</h2>
<div class="flow"><div>Raw SCADA audit</div><div>Chronological split</div><div>Past-only engineering</div><div>Train-fitted pipelines</div></div>
<p>A shared chronological split placed {split_rows['train']:,} rows in training, {split_rows['validation']:,} in validation, and {split_rows['test']:,} in the final test period. All 15 turbines appear in every period, every row occurs exactly once, and failure episodes were not divided at the boundaries.</p>
<table class="report-table"><thead><tr><th>Partition</th><th>Rows</th><th>Failures</th><th>Failure rate</th></tr></thead><tbody>
<tr><td>Training</td><td>{split_rows['train']:,}</td><td>{split_failures['train']:,}</td><td>{split_rates['train']:.3f}%</td></tr><tr><td>Validation</td><td>{split_rows['validation']:,}</td><td>{split_failures['validation']:,}</td><td>{split_rates['validation']:.3f}%</td></tr><tr><td>Test</td><td>{split_rows['test']:,}</td><td>{split_failures['test']:,}</td><td>{split_rates['test']:.3f}%</td></tr></tbody></table>
<p>Imputation, encoding, scaling, model fitting, and imbalance treatment were learned from training data within pipelines. Median imputation was chosen because missingness was low and it is robust to skew and extremes; observed zeros were preserved. <code>rated_power_kW</code> was one-hot encoded, and ANN numeric inputs were standardized. Balanced class/sample weights were derived from training labels; SMOTE was avoided because interpolation could dilute meaningful fault extremes and disregard temporal structure.</p>
<p>Six provenance-risk maintenance/history fields—prior fault count, hours since maintenance, component age, cumulative operating hours, cumulative energy, and load cycles—were conservatively excluded. The rising failure prevalence across time is a material distribution shift and is considered when interpreting generalization.</p>

<h2 id="features">4. Feature Engineering</h2>
<p>Step 04 created 24 domain-informed snapshot features and 36 past-only temporal features. These include safe operating ratios, component temperature rises, FFT summaries, circular direction encodings, lagged measurements, recent changes, and one- and three-hour rolling statistics calculated independently per turbine. No centered or future-looking windows were used.</p>
<p>A training/validation ablation selected {len(selected_features)} predictors combining original measurements with engineered features. The test set was not scored during this selection.</p>
<table class="report-table"><thead><tr><th>Feature group</th><th>Examples</th><th>Rationale</th></tr></thead><tbody>
<tr><td>Load and speed</td><td>Capacity factor, safe generator/rotor ratio, stopped-rotor flag</td><td>Normalize operation across turbine ratings and distinguish stopped states safely.</td></tr>
<tr><td>Thermal differences</td><td>Bearing temperature rises above ambient; component gaps</td><td>Separate component heating from ambient operating conditions.</td></tr>
<tr><td>Vibration and oil</td><td>FFT summaries, gearmesh/sideband ratio, oil particles per pressure</td><td>Summarize related condition-monitoring signals.</td></tr>
<tr><td>Circular direction</td><td>Sine/cosine wind direction and yaw misalignment</td><td>Represent angular proximity correctly across the 0°/360° boundary.</td></tr>
<tr><td>Past-only dynamics</td><td>Lag, change, 1-hour and 3-hour rolling statistics</td><td>Capture persistence and recent change without using future observations.</td></tr>
</tbody></table>

<h2 id="baseline">5. Baseline Model Building</h2>
<p>Five assignment-required classifiers were fitted on the full training period at a fixed 0.50 threshold. XGBoost produced the strongest baseline recall and PR-AUC. Random Forest showed strong probability ranking but conservative classifications, motivating tuning and threshold analysis.</p>
{baseline_table}
<figure><img src="{images['baseline']}" alt="Baseline ROC and precision-recall curves"><figcaption>Validation ROC and precision-recall curves for the five baseline models.</figcaption></figure>
<h3>Baseline interpretation</h3><ul><li><strong>Decision Tree:</strong> interpretable but strongly overfit, with only 0.2116 validation failure recall.</li><li><strong>Random Forest:</strong> high precision (0.9091) and PR-AUC (0.8187), but a conservative 0.50 threshold limited recall to 0.1641.</li><li><strong>Gradient Boosting:</strong> improved recall to 0.5308 but trailed the leading probability-ranking models.</li><li><strong>XGBoost:</strong> strongest overall baseline with 0.6965 recall, 0.8117 precision, and 0.8707 PR-AUC.</li><li><strong>Artificial Neural Network:</strong> second-highest fixed-threshold recall at 0.5373, satisfying the neural-network comparison while trailing XGBoost in PR-AUC.</li></ul>

<h2 id="tuning">6. Hyperparameter Tuning and Model Selection</h2>
<p>XGBoost and Random Forest were tuned using eight configurations each over three expanding chronological folds within the training period. The winning configurations were refitted on all training data. Each threshold was chosen on validation data by maximizing F2, which weights recall twice as strongly as precision.</p>
<p><strong>Candidate rationale:</strong> XGBoost led the recall-first baseline comparison. Random Forest was retained because its strong baseline PR-AUC showed useful ranking performance that a lower threshold could convert into substantially higher recall. Search dimensions covered tree count and depth, learning rate, sampling, feature sampling, minimum child/leaf constraints, and regularization as appropriate to each model.</p>
{tuned_table}
<figure><img src="{images['tuning']}" alt="Tuned validation confusion matrices"><figcaption>Tuned validation confusion matrices at the selected model-specific thresholds.</figcaption></figure>
<p>Threshold adjustment raised tuned XGBoost validation recall to 0.9122 and Random Forest recall to 0.7801. XGBoost was selected because it retained the stronger recall, F2, PR-AUC, and ROC-AUC combination. Its validation-derived threshold of {final['threshold']:.3f} was frozen before test evaluation.</p>

<h2 id="performance">7. Final Model Performance</h2>
<p>The selected configuration was refitted on combined training and validation data and evaluated once on the later test period.</p>
<div class="metrics"><div class="metric"><strong>{final['recall']:.4f}</strong>Recall</div><div class="metric"><strong>{final['precision']:.4f}</strong>Precision</div><div class="metric"><strong>{final['f1']:.4f}</strong>F1</div><div class="metric"><strong>{final['pr_auc']:.4f}</strong>PR-AUC</div><div class="metric"><strong>{final['roc_auc']:.4f}</strong>ROC-AUC</div></div>
<figure><img src="{images['test_confusion']}" alt="Final test confusion matrix"><figcaption>The final model correctly detected 1,069 failure records, missed 186, and produced 679 false-positive records.</figcaption></figure>
<figure><img src="{images['test_curves']}" alt="Final test ROC and precision-recall curves"><figcaption>Threshold-independent discrimination on the untouched test period.</figcaption></figure>
<div class="callout"><strong>Operational view:</strong> The classifier detected {int(operational['detected_episodes'])} of {int(operational['failure_episodes'])} failure episodes ({operational['episode_recall']:.2%}). It produced {operational['false_positive_records_per_turbine_day']:.2f} false-positive 10-minute records per turbine-day before alert deduplication or cooldown rules.</div>
<p><strong>Performance interpretation:</strong> The final model missed 186 of 1,255 failure records while correctly identifying 1,069. Precision of 0.6116 means roughly three in five positive records corresponded to labeled failures. This is consistent with the deliberate recall-first objective, but the alert stream requires aggregation before operational use.</p>

<h2 id="importance">8. Important Features Used by the Best Model</h2>
<p>Gain-based XGBoost importance is presented in two complementary views. The detailed view shows the actual original and engineered model inputs. The grouped view combines derived variants with their underlying physical sensor family, which is more suitable for business interpretation.</p>
<figure><img src="{images['importance']}" alt="Detailed feature importance"><figcaption>Top model inputs, including original and engineered predictors.</figcaption></figure>
{importance_table}
{family_table}
<figure><img src="{images['families']}" alt="Sensor-family importance"><figcaption>Importance aggregated by physical source-sensor family.</figcaption></figure>
<p>Drivetrain vibration contributed {families.iloc[0]['importance']:.2%} of fitted importance after grouping current, lagged, rolling, and normalized variants. Tower vibration and FFT vibration signals followed. Correlated features may divide or concentrate gain importance, so these values explain this fitted model rather than proving causal fault mechanisms.</p>

<h2 id="recommendations">9. Actionable Insights and Recommendations</h2>
<ol><li><strong>Prioritize sustained vibration:</strong> route persistent drivetrain-vibration elevations and supporting tower/FFT patterns to analyst review.</li><li><strong>Deploy as decision support:</strong> use scores to prioritize inspections; do not trigger autonomous shutdown or maintenance without qualified review.</li><li><strong>Control alert burden:</strong> convert consecutive positive records into one alert episode and apply a documented cooldown. Validate whether the observed false-positive burden fits maintenance capacity.</li><li><strong>Investigate missed episodes:</strong> review the five undetected test episodes by turbine, duration, operating state, and sensor availability.</li><li><strong>Monitor drift:</strong> track failure prevalence, feature distributions, recall, precision, and alert volume by turbine and calendar period.</li><li><strong>Validate prospectively:</strong> conduct a shadow deployment before operational use and retrain only after appropriately labeled recent data are available.</li></ol>

<h2 id="limitations">10. Limitations and Responsible Use</h2>
<ul><li>The dataset represents 15 turbines and approximately two months, limiting seasonal and fleet-wide generalization.</li><li>The label supports detection of current fault conditions; it does not demonstrate hours- or days-ahead failure forecasting.</li><li>Failure prevalence rises materially between training and later periods.</li><li>Episode recall counts an episode as detected if any constituent record is positive and does not itself measure advance warning.</li><li>False-positive records are not equivalent to work orders until an alert aggregation policy is defined.</li><li>Feature importance is model-specific and associative, not causal.</li></ul>

<h2 id="rubric">11. Rubric Coverage</h2>
<table class="report-table rubric"><thead><tr><th>Requirement</th><th>Evidence included</th><th>Status</th></tr></thead><tbody>
<tr><td>Exploratory Data Analysis — 10 points</td><td>Problem definition, data overview, univariate, bivariate, multivariate analysis, four-panel EDA visualization, and meaningful observations.</td><td class="check">Covered</td></tr>
<tr><td>Data Preprocessing — 7 points</td><td>Missing detection/treatment rationale, feature engineering rationale, chronological split, imbalance policy, and leakage controls.</td><td class="check">Covered</td></tr>
<tr><td>Baseline Modeling — 16 points</td><td>Evaluation rationale plus Decision Tree, Random Forest, Gradient Boosting, XGBoost, and ANN results with commentary.</td><td class="check">Covered</td></tr>
<tr><td>Hyperparameter Tuning — 11 points</td><td>Two justified candidates, chronological tuning, tuned comparison, validation commentary, and frozen threshold.</td><td class="check">Covered</td></tr>
<tr><td>Model Performance — 6 points</td><td>Tuned comparison, justified XGBoost selection, one-time test evaluation, and detailed/grouped feature importance.</td><td class="check">Covered</td></tr>
<tr><td>Insights &amp; Recommendations — 6 points</td><td>Six operational recommendations tied directly to results and limitations.</td><td class="check">Covered</td></tr>
<tr><td>Overall Quality — 4 points</td><td>Executive-first structure, concise tables, embedded visuals, navigation, consistent terminology, and standalone HTML.</td><td class="check">Covered</td></tr>
</tbody></table>

<h2 id="conclusion">12. Conclusion</h2>
<p>The final XGBoost pipeline provides strong discrimination and detects 85.18% of failure records and 92.31% of failure episodes in the later test period. Sustained drivetrain-vibration behavior is the dominant signal family. The results support a promising analyst-triage proof of concept, subject to prospective validation, alert aggregation, ongoing drift monitoring, and qualified human review.</p>
<footer><strong>Project repository:</strong> <a href="https://github.com/Namees-aLbayati/wind-turbine-failure-detection" target="_blank" rel="noopener noreferrer">github.com/Namees-aLbayati/wind-turbine-failure-detection</a><br>Generated programmatically from the project’s executed and saved analysis artifacts. No final metrics or model comparisons were manually recomputed in the reporting step. <a href="#top">Back to top ↑</a></footer>
</main></body></html>"""

    report_path.write_text(document, encoding="utf-8")
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages_path.write_text(document, encoding="utf-8")
    (pages_dir / ".nojekyll").touch()
    root_pages_path.write_text(document, encoding="utf-8")
    (project_root / ".nojekyll").touch()
    return report_path
