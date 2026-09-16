"""Render notebook recaps from the executed CSV/JSON artifacts."""
from pathlib import Path
import json
import nbformat
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reports/model_results'

def refresh(name):
    path = ROOT / 'notebooks' / name
    notebook = nbformat.read(path, 4)
    if name.startswith('04_'):
        a = pd.read_csv(RESULTS/'04_feature_ablation.csv').iloc[0]
        metadata = {s: pd.read_csv(ROOT/f'data/processed/03_{s}_metadata.csv') for s in ['train','validation','test']}
        bullets = [f"Exported {len(metadata['train']):,} training, {len(metadata['validation']):,} validation and {len(metadata['test']):,} test rows on the corrected split.",
            'Three complete test turbines are absent from development data.',
            'Created 24 snapshot and 36 past-only per-turbine temporal features.',
            f"Validation selection retained {a.feature_group} with {int(a.feature_count)} predictors.",
            f"The selected feature group achieved validation recall {a.recall_at_0_5:.4f} and PR-AUC {a.pr_auc:.4f}.",
            'Test scores were not used to choose the feature group.']
    elif name.startswith('05_baseline'):
        b=pd.read_csv(RESULTS/'05_baseline_metrics.csv');best=b.sort_values('validation_pr_auc',ascending=False).iloc[0]
        ann=b[b.model.eq('Artificial Neural Network')].iloc[0]
        bullets=['Five baseline pipelines completed on the corrected grouped split.',
            f"{best.model} led baseline validation PR-AUC at {best.validation_pr_auc:.4f}.",
            f"The scikit-learn ANN reached recall {ann.validation_recall:.4f}, precision {ann.validation_precision:.4f} and PR-AUC {ann.validation_pr_auc:.4f} at threshold 0.50.",
            'XGBoost and Random Forest are the prespecified tree tuning candidates.',
            'Notebook 05b supplies five Keras comparisons and a neural candidate for cross-family selection.',
            'Baselines saved fitted pipelines and validation diagnostics without scoring test labels.']
    elif name.startswith('06_'):
        t=pd.read_csv(RESULTS/'06_tuned_validation_metrics.csv');best=t.iloc[0]
        bullets=['Each tree family evaluated eight configurations over three training-only chronological folds.',
            'XGBoost imbalance weighting was recomputed within each training fold.',
            'Two tuned tree pipelines and the validation-selected neural checkpoint were compared on the same validation rows.',
            f"{best.model} led the documented PR-AUC-first selection with PR-AUC {best.pr_auc:.4f} and F2 {best.f2:.4f}.",
            f"Its frozen threshold is {best.threshold:.3f}, with validation recall {best.recall:.4f} and precision {best.precision:.4f}.",
            'The selected checkpoint and artifact fingerprints were saved; test outcomes did not influence selection.']
    elif name.startswith('07_'):
        f=pd.read_csv(RESULTS/'07_final_test_metrics.csv').iloc[0];o=pd.read_csv(RESULTS/'07_final_test_operational_metrics.csv').iloc[0]
        i=pd.read_csv(RESULTS/'07_final_feature_importance.csv').iloc[0]
        bullets=[f"Evaluated the frozen {f.model} checkpoint at threshold {f.threshold:.3f} without refitting.",
            f"The held-out-turbine test set contains {int(f.rows):,} rows and {int(f.failure_count):,} failures.",
            f"Test recall was {f.recall:.4f}, precision {f.precision:.4f}, F1 {f.f1:.4f}, PR-AUC {f.pr_auc:.4f}, and ROC-AUC {f.roc_auc:.4f}.",
            f"Detected {int(o.detected_episodes)} of {int(o.failure_episodes)} episodes; false-positive burden was {o.false_positive_records_per_turbine_day:.4f} records per turbine-day.",
            f"The leading input was {i.feature}, using {i.method.lower()}.",
            'Earlier exploratory test exposure means these results are not fresh independent confirmation.',
            'Generic final-model artifacts and result fingerprints support reporting regardless of the winning family.']
    elif name.startswith('08_'):
        f=pd.read_csv(RESULTS/'07_final_test_metrics.csv').iloc[0]
        bullets=['Generated standalone HTML from verified selection and result artifacts.',
            'Included the five Keras experiments alongside traditional baselines and tuned candidates.',
            f"Reported selected {f.model} test recall {f.recall:.4f} and precision {f.precision:.4f} consistently.",
            'Documented grouped test separation, temporal validation, representation differences and earlier test exposure.',
            'Embedded figures and validated required sections, HTML structure and size.',
            'Updated local report and site files; no remote publication was performed.']
    else:
        return
    notebook.cells[-1].source='## Key Findings Recap\n\n'+'\n'.join('- '+x for x in bullets)
    nbformat.write(notebook,path)

if __name__=='__main__':
    for name in ['04_feature_engineering.ipynb','05_baseline_modeling.ipynb','06_hyperparameter_tuning.ipynb','07_final_model_evaluation.ipynb','08_final_reporting.ipynb']:
        refresh(name)
