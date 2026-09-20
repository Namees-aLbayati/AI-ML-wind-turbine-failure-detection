"""Attach interpretation of the recorded model comparisons without retraining."""
from pathlib import Path
import json
import pandas as pd
import nbformat as nb


def add_model_selection_interpretation(notebook):
    root = Path(__file__).resolve().parents[1]
    results = pd.read_csv(root / 'models/template_aligned/model_comparison.csv').set_index(['model', 'split'])
    tuned_names = list(dict.fromkeys(name for name, _ in results.index if name.endswith(' tuned')))
    rows = []
    explanations = []
    for tuned in tuned_names:
        name = tuned.removesuffix(' tuned')
        train, valid = results.loc[(name, 'train')], results.loc[(name, 'validation')]
        train_t, valid_t = results.loc[(tuned, 'train')], results.loc[(tuned, 'validation')]
        rows.append(f"| {name} | {train.Recall:.4f} → {train_t.Recall:.4f} | {train.PR_AUC:.4f} → {train_t.PR_AUC:.4f} | {train.PR_AUC-valid.PR_AUC:.4f} → {train_t.PR_AUC-valid_t.PR_AUC:.4f} |")
        explanations.append(f"**{name}:** Training recall changed by {train_t.Recall-train.Recall:+.4f} and training PR-AUC by {train_t.PR_AUC-train.PR_AUC:+.4f}. Validation recall changed by {valid_t.Recall-valid.Recall:+.4f} and validation PR-AUC by {valid_t.PR_AUC-valid.PR_AUC:+.4f}.")
    training = ('**Training performance before and after tuning**\n\n'
                '| Model | Training recall: baseline → tuned | Training PR-AUC: baseline → tuned | Train−validation PR-AUC gap: baseline → tuned |\n'
                '| --- | --- | --- | --- |\n' + '\n'.join(rows) + '\n\n' + '\n\n'.join(explanations))
    generalization = ('**Interpretation of generalization**\n\n'
        'XGBoost and the neural network have lower training PR-AUC after tuning but higher validation recall and PR-AUC. '
        'Their smaller train–validation gaps are consistent with reduced overfitting on this validation split. '
        'Lower training performance is therefore not automatically a worse result. '
        'Gradient Boosting also has a smaller gap, but both validation recall and PR-AUC decline; its tuning does not improve the recall-first objective. '
        'A smaller gap alone does not establish a better model.\n\n'
        'The tuned ANN improves validation recall while reducing validation precision relative to its baseline. '
        'It wins under the declared recall-first rule, not because it dominates every metric. '
        'Repeated use of this validation set means these comparisons are not independent confirmation.')
    importance = ('**Feature-importance interpretation**\n\n'
        'For the selected ANN, importance measures the decrease in validation PR-AUC when a feature is shuffled, '
        'averaged across three repeats on up to 3,000 validation records. This evaluates reliance on features outside the training data. '
        'Three repeats provide an exploratory estimate; additional repeats could assess its stability. '
        'Correlated features can share or mask importance, and importance does not establish causation.')
    metrics = json.loads((root / 'models/template_aligned/final_metrics.json').read_text())
    tradeoff = ('**Selected-model tradeoff**\n\n'
        f"The validation-selected {metrics['model']} achieved test recall of {metrics['Recall']:.2%} and precision of {metrics['Precision']:.2%} "
        f"at threshold {metrics['threshold']:.2f}. It detects nearly all labeled fault records, but {1-metrics['Precision']:.2%} of flagged records are false positives. "
        'This supports its role as a recall-focused triage candidate; alerts require investigation, grouping, and workload assessment before operational use. '
        'These are record-level results, not counts of independent breakdowns or maintenance visits. Earlier test exposure limits independent confirmation.')
    additions = {225: training, 227: generalization, 232: importance, 235: tradeoff}
    cells = []
    for cell in notebook.cells:
        if cell.metadata.get('model_selection_interpretation'):
            continue
        cells.append(cell)
        idx = cell.metadata.get('template_cell')
        if idx in additions:
            note = nb.v4.new_markdown_cell(additions[idx])
            note.metadata['model_selection_interpretation'] = True
            cells.append(note)
    notebook.cells = cells
    return notebook
