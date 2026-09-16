"""Validate current pipeline artifacts without retraining or rescoring models."""
from pathlib import Path
import json
import sys
import nbformat
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.metrics import recall_score, precision_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.modeling.artifacts import check_splits, verify_manifest


def main():
    frames=check_splits(ROOT)
    verify_manifest(ROOT,ROOT/'models/06_selection_manifest.json')
    verify_manifest(ROOT,ROOT/'models/07_results_manifest.json')
    result_dir=ROOT/'reports/model_results'
    comparison=pd.read_csv(result_dir/'06_tuned_validation_metrics.csv')
    winner=comparison.sort_values(['pr_auc','f2','recall'],ascending=False).iloc[0]
    selection=json.loads((ROOT/'models/06_selected_threshold.json').read_text())
    assert selection['model']==winner.model
    assert np.isclose(selection['threshold'],winner.threshold)
    final=pd.read_csv(result_dir/'07_final_test_metrics.csv').iloc[0]
    predictions=pd.read_csv(result_dir/'07_final_test_predictions.csv')
    assert final.model==selection['model']
    assert np.isclose(final.threshold,selection['threshold'])
    assert set(predictions.source_index)==set(frames['test'].source_index)
    assert len(predictions)==len(frames['test'])==int(final.rows)
    assert np.isclose(final.recall,recall_score(predictions.failure,predictions.prediction))
    assert np.isclose(final.precision,precision_score(predictions.failure,predictions.prediction))
    assert np.isclose(final.pr_auc,average_precision_score(predictions.failure,predictions.failure_probability))
    expected=(predictions.failure_probability>=selection['threshold']).astype(int)
    assert np.array_equal(expected,predictions.prediction)
    for name in ['01_data_understanding','02_eda','03_preprocessing','04_feature_engineering','05_baseline_modeling','05b_neural_networks','06_hyperparameter_tuning','07_final_model_evaluation','08_final_reporting']:
        notebook=nbformat.read(ROOT/f'notebooks/{name}.ipynb',4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type=='code':
                assert cell.execution_count is not None
                assert not any(o.output_type=='error' for o in cell.outputs)
        assert notebook.cells[-1].source.startswith('## Key Findings Recap')
        assert 'Pending' not in notebook.cells[-1].source
    report=(ROOT/'reports/Wind_Turbine_Failure_Detection_Report.html').read_text()
    assert 'Neural Network Comparison' in report
    assert 'not fresh independent confirmation' in report
    assert f'{final.recall:.2%}' in report
    assert report.count('data:image/png;base64,')>=8
    assert report==(ROOT/'docs/index.html').read_text()==(ROOT/'index.html').read_text()
    combined = nbformat.read(ROOT/'notebooks/AIML_Project_1_Full_Code_Notebook_Completed.ipynb', 4)
    html = BeautifulSoup((ROOT/'reports/AIML_Project_1_Full_Code_Notebook_Completed.html').read_text(), 'html.parser')
    assert len(html.select('div.jp-Cell')) == len(combined.cells)
    for cell in combined.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None
            assert not any(o.output_type == 'error' for o in cell.outputs)
    assert len(html.select('section[id^="section-"]')) == 13
    assert all(img.get('src', '').startswith('data:') for img in html.find_all('img'))
    assert 'do not isolate optimizer effects' in html.get_text()
    print('Passed: split identity, model/result fingerprints, validation winner, frozen threshold, saved test metrics, executed notebooks, and report consistency.')

if __name__=='__main__':
    main()
