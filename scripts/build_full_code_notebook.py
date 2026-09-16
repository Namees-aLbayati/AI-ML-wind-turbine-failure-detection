"""Assemble dependency-ordered source notebooks and their executed outputs."""
from pathlib import Path
import nbformat as nb
from nbconvert import HTMLExporter
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['01_data_understanding.ipynb','02_eda.ipynb','03_preprocessing.ipynb',
           '04_feature_engineering.ipynb','05_baseline_modeling.ipynb','05b_neural_networks.ipynb',
           '06_hyperparameter_tuning.ipynb','07_final_model_evaluation.ipynb','08_final_reporting.ipynb']

def main():
    combined = nb.v4.new_notebook()
    combined.metadata.kernelspec = {'display_name':'Python 3','language':'python','name':'python3'}
    combined.cells = [nb.v4.new_markdown_cell('''# Wind Turbine Failure Detection — Full Code

This document assembles the development notebooks in execution order and retains their recorded outputs. Install `requirements-neural.txt` and run from the repository root or its notebooks directory. Shared implementation sources are included in the appendix for review and live under `src/`.

The corrected protocol holds out three complete turbines and uses later development observations for validation. Five Keras experiments supplement traditional baselines. Stage 06 ranks complete pipelines using validation PR-AUC and selects thresholds with validation F2. Stage 07 evaluates the frozen winner without refitting. Earlier test exposure is disclosed; these results are not fresh independent confirmation.

Recorded outputs were produced in the individual notebook kernels. This consolidated artifact preserves the original dependency order; it does not reorder feature-importance or evaluation cells.''')]
    for name in SOURCES:
        source = nb.read(ROOT/'notebooks'/name,4)
        combined.cells.append(nb.v4.new_markdown_cell(f'---\n\nSource: `{name}`'))
        combined.cells.extend(source.cells)
    combined.cells.append(nb.v4.new_markdown_cell('# Shared Implementation Sources'))
    for path in ['src/modeling/neural_networks.py','src/modeling/artifacts.py','src/reporting/build_report.py']:
        combined.cells.append(nb.v4.new_markdown_cell(f'## `{path}`\n\n```python\n{(ROOT/path).read_text()}\n```'))
    combined.cells.append(nb.v4.new_markdown_cell('''## Key Findings Recap

- Source notebooks are assembled in dependency order, including stage 05b.
- Grouped test separation and temporal validation are used downstream.
- Model selection depends on validation results rather than a hard-coded model family.
- Final checkpoint and result fingerprints guard against stale inputs.
- Final reporting contains regenerated metrics and disclosed evaluation limitations.
- This document preserves recorded source outputs; reruns require the project data and dependencies.'''))
    for index, cell in enumerate(combined.cells):
        cell.id = f"full-code-{index:04d}"
    nb.validate(combined)
    for cell in combined.cells:
        if cell.cell_type=='code':
            compile(cell.source,'combined-cell','exec')
            if any(o.output_type=='error' for o in cell.outputs):
                raise RuntimeError('Source notebook contains an execution error.')
    output=ROOT/'notebooks/AIML_Project_1_Full_Code_Notebook_Completed.ipynb'
    nb.write(combined,output)
    exporter=HTMLExporter()
    exporter.embed_images=True
    document,_=exporter.from_notebook_node(combined,resources={'metadata':{'path':str(ROOT/'notebooks')}})
    document = '\n'.join(line.rstrip() for line in document.splitlines()) + '\n'
    (ROOT/'reports/AIML_Project_1_Full_Code_Notebook_Completed.html').write_text(document)
    print('Built full-code notebook and HTML with neural-network stage included.')

if __name__=='__main__':
    main()
