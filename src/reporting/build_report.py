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
    from src.modeling.artifacts import verify_manifest
    import json
    root = Path(project_root).resolve()
    results = root / 'reports/model_results'
    figures = root / 'reports/figures'
    verify_manifest(root, root / 'models/06_selection_manifest.json')
    verify_manifest(root, root / 'models/07_results_manifest.json')
    raw = pd.read_csv(root / 'data/raw/wind_turbine_detection.csv')
    baseline = pd.read_csv(results / '05_baseline_metrics.csv')
    neural = pd.read_csv(root / 'models/neural_network_comparison/comparison.csv')
    tuned = pd.read_csv(results / '06_tuned_validation_metrics.csv')
    final = pd.read_csv(results / '07_final_test_metrics.csv').iloc[0]
    operational = pd.read_csv(results / '07_final_test_operational_metrics.csv').iloc[0]
    importance = pd.read_csv(results / '07_final_importance_interpretation.csv')
    families = pd.read_csv(results / '07_final_sensor_family_importance.csv')
    ablation = pd.read_csv(results / '04_feature_ablation.csv')
    splits = []
    for split in ['train', 'validation', 'test']:
        frame = pd.read_csv(root / f'data/processed/03_{split}_metadata.csv')
        splits.append(dict(split=split, rows=len(frame), turbines=frame.turbine_id.nunique(), failures=int(frame.failure.sum()), failure_rate=frame.failure.mean()))
    _create_eda_summary(raw, figures / '08_report_eda_summary.png')
    def table(frame):
        return frame.round(4).to_html(index=False, border=0, classes='report-table')
    def figure(path, caption):
        return f'<figure><img src="{_image_data_uri(path)}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'
    name = html.escape(str(final['model']))
    missed = int(final.failure_count * (1-final.recall) + 0.5)
    sections = [
        ('Executive Summary', f'<p>The validation-selected <strong>{name}</strong> achieved <strong>{final.recall:.2%} failure recall</strong> and <strong>{final.precision:.2%} precision</strong> on {int(final.rows):,} records from three held-out turbines. The frozen threshold is {final.threshold:.3f}. False-positive burden is {operational.false_positive_records_per_turbine_day:.2f} records per turbine-day.</p>'),
        ('Business Problem and Objective', '<p>Identify current drivetrain faults in 10-minute SCADA records to support inspection prioritization. The label measures current fault detection, not advance forecasting. Missed failures motivate recall-sensitive evaluation; precision and alert burden constrain practical usefulness.</p>'),
        ('Data Overview and Exploratory Analysis', f'<p>The dataset contains {len(raw):,} rows, {raw.shape[1]} columns, {raw.turbine_id.nunique()} turbines and {int(raw.failure.sum()):,} positive records. EDA reviews missingness, distributions, condition associations and temporal structure. Outliers may carry fault information and are retained.</p>' + figure(figures / '08_report_eda_summary.png', 'Target balance, turbine rates, vibration and correlations.')),
        ('Preprocessing and Leakage Controls', '<p>Three complete turbines form the test partition. Within the remaining 12 turbines, earlier rows train the models and later rows validate them, with failure episodes kept intact at the boundary. Test timestamps can overlap development timestamps because turbine identity provides the outer separation. Metadata and artifact fingerprints enforce consistent inputs.</p>' + table(pd.DataFrame(splits)) + '<p>Preprocessors fit on training data. The neural branch uses median imputation, Yeo–Johnson correction, scaling, FFT PCA and categorical encoding; tree pipelines fit their own imputers and encoders. Six history fields with uncertain provenance are excluded. Neural class weights and tree imbalance treatment use training labels; XGBoost recomputes its weight within each cross-validation fit.</p>'),
        ('Feature Engineering', '<p>Tree candidates use selected original, snapshot and past-only per-turbine features. Lagged and rolling features use preceding sensor observations, never future measurements. The neural experiments use the 25-input stage-03 representation. Cross-family comparisons therefore compare complete pipelines, not architecture alone. Feature-group selection uses validation data, creating selection uncertainty on that reused validation set.</p>' + table(ablation)),
        ('Baseline Model Comparison', '<p>Five traditional baselines use the corrected partitions and a 0.50 threshold. XGBoost and Random Forest are prespecified tree tuning candidates; the neural comparison adds an independently configured candidate.</p>' + table(baseline[['model','validation_recall','validation_precision','validation_f1','validation_pr_auc','validation_roc_auc']]) + figure(figures / '05_baseline_roc_pr_curves.png', 'Baseline validation ROC and precision-recall curves.')),
        ('Neural Network Comparison', '<p>Five Keras configurations compare a linear baseline, 128/64 SGD, 128/64 Adam, batch normalization with 50% dropout, and a 64/32 Adam control. All use the same training-derived weighting, up to 30 epochs and validation-loss early stopping with patience five. Best validation-loss weights are restored. Validation PR-AUC selects the neural candidate; larger layers and dropout are not assumed to improve performance.</p>' + table(neural) + figure(root / 'models/neural_network_comparison/learning_curves.png', 'Training and validation loss; weighting and dropout make absolute loss levels non-comparable.')),
        ('Hyperparameter Tuning and Threshold Selection', '<p>Each tree family evaluates eight configurations on three training-only expanding chronological folds. Inner search ranks recall then PR-AUC. The two tuned pipelines and the selected neural checkpoint are compared on identical validation source rows. Overall selection ranks validation PR-AUC, then F2 and recall. Each threshold maximizes validation F2, which emphasizes recall. No test score enters these rules.</p>' + table(tuned) + figure(figures / '06_tuned_validation_confusion_matrices.png', 'All final candidates at their validation-selected thresholds.')),
        ('Final Test Performance', f'<p>The selected {name} training-fitted checkpoint is evaluated without refitting, preserving the probability scale used to select threshold {final.threshold:.3f}. It missed {missed:,} labeled failure records. Final results below supersede the earlier chronological-split report.</p>' + table(pd.DataFrame([final])) + table(pd.DataFrame([operational])) + figure(figures / '07_final_test_confusion_matrix.png', 'Selected model: held-out-turbine confusion matrix.') + figure(figures / '07_final_test_roc_pr_curves.png', 'Held-out-turbine ROC and precision-recall curves.')),
        ('Feature Importance', '<p>Tree winners use fitted tree importance; neural winners use validation-sample permutation PR-AUC decrease. These measures are model-specific associations. Permutation values are not normalized percentages, and PCA components do not identify a single physical sensor.</p>' + table(importance.head(20)) + table(families.head(10)) + figure(figures / '07_final_feature_importance.png', 'Importance of the selected model inputs.') + figure(figures / '07_final_sensor_family_importance.png', 'Importance grouped by source-sensor family.')),
        ('Actionable Insights and Recommendations', f'<p>The model detected {int(operational.detected_episodes)} of {int(operational.failure_episodes)} fault episodes. Review missed episodes and the leading model inputs with domain specialists. Aggregate consecutive alerts and test cooldown rules before translating positive records into work orders. Use predictions for analyst triage and validate prospectively before operational deployment.</p>'),
        ('Limitations and Responsible Use', '<p>Only 15 turbines and roughly two months are represented. Three test turbines provide limited evidence for fleet-wide generalization. Earlier exploratory test outcomes have already been viewed, so the regenerated test results are not fresh independent confirmation. Validation is reused for feature selection, model comparisons and threshold choice. Temporal CV does not by itself measure unseen-turbine generalization. Prevalence differs between partitions. Reported episode detection is not advance warning, and false-positive records are not equivalent to distinct alarms. No performance improvement over the old split is claimed.</p>'),
        ('Conclusion', f'<p>{name} was selected using the documented validation rule. Recall of {final.recall:.2%} and precision of {final.precision:.2%} quantify its current detection tradeoff on held-out turbines. The next step is prospective validation and alert-policy evaluation.</p>'),
    ]
    contents = ''.join(f'<li><a href="#section-{i}">{title}</a></li>' for i,(title,_) in enumerate(sections))
    body = ''.join(f'<section id="section-{i}"><h2>{title}</h2>{content}</section>' for i,(title,content) in enumerate(sections))
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Wind Turbine Failure Detection</title><style>
body{margin:0;background:#eef3f6;color:#23384b;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:30px auto;background:white;padding:40px;border-radius:14px}h1,h2{color:#173b55}h2{border-bottom:2px solid #2a9d8f;padding-top:20px}section{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border:1px solid #d4e0e6;text-align:left}th{background:#173b55;color:white}tr:nth-child(even){background:#f2f7f9}img{max-width:100%;height:auto}figure{margin:24px 0}figcaption{color:#516777}a{color:#146b86}@media(max-width:700px){main{padding:16px;margin:0}}
</style></head><body><main><h1>Wind Turbine Failure Detection</h1><p>Corrected grouped split · Neural and tree pipeline comparison · Executed results</p><nav><ol>''' + contents + '</ol></nav>' + body + '<footer>Generated from verified local artifacts. Install requirements-neural.txt and run notebooks in the documented dependency order.</footer></main></body></html>'
    report_path = root / 'reports/Wind_Turbine_Failure_Detection_Report.html'
    for path in [report_path, root / 'docs/index.html', root / 'index.html']:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding='utf-8')
    return report_path
