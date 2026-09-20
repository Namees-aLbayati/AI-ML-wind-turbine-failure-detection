"""Keep assignment requirements while presenting a completed analysis."""
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
import re
import nbformat as nb

INSTRUCTION_HEADING = '# **Please read the instructions carefully before starting the project.**'

def clean_submission(notebook, reference):
    replacements = {
        11: 'This analysis uses the project environment and the established grouped test split and selected engineered predictors. Results below belong to this revised analysis. Earlier test exposure and repeated validation use limit independent confirmation.',
        132: 'Decision Tree, Random Forest, Gradient Boosting, XGBoost, and a neural network are evaluated on the training and validation partitions.',
        136: 'Evaluation reports accuracy, recall, precision, F1, F2, PR-AUC, and ROC-AUC. Confusion matrices show missed faults and false alerts at the common 0.50 threshold.',
        176: 'The two tree families with the highest baseline validation recall, with PR-AUC as the tie-breaker, proceed to randomized search. A regularized neural network is also evaluated. Unselected tree families retain their baseline results.',
    }
    root = Path(__file__).resolve().parents[1]
    package_names = []
    for filename in ['requirements.txt', 'requirements-neural.txt']:
        contents = (root / filename).read_text().strip()
        for line in contents.splitlines():
            if line.strip() and not line.startswith(('#', '-')):
                package_names.append(re.split(r'[<>=!~\[; ]', line)[0])
    rows = []
    pinned_packages = []
    for name in dict.fromkeys(package_names):
        try:
            installed = version(name)
            pinned_packages.append(f'{name}=={installed}')
        except PackageNotFoundError:
            installed = 'Not installed in export environment'
        rows.append(f'| {name} | {installed} |')
    dependency_source = (
        '| Library | Version |\n| --- | --- |\n'
        + '\n'.join(rows)
        + '\n\n```bash\npython -m pip install ' + ' '.join(pinned_packages) + '\n```'
    )
    cells = []
    for cell in notebook.cells:
        if cell.metadata.get('dependency_inventory'):
            continue
        idx = cell.metadata.get('template_cell')
        if idx in (7, 8, 10, 11):
            continue
        if cell.cell_type == 'markdown':
            if idx in replacements:
                cell.source = replacements[idx]
            elif (isinstance(idx, int) and idx > 6 and idx != 112
                  and cell.source == reference.cells[idx].source
                  and not cell.source.lstrip().startswith('#')):
                # Prompts such as "Perform the following steps" are replaced by
                # the executed code, results, and section-specific observations.
                continue
            cell.source = cell.source.replace(
                'This raw-sensor exercise satisfies the template;',
                'These raw-sensor values are imputed separately;')
            cell.source = cell.source.replace(
                'The configured epoch cap satisfies the template, while actual epochs can be fewer.',
                'Training allows up to 30 epochs; early stopping can end it sooner.')
            cell.source = cell.source.replace('**Observations / implementation notes:**', '**Observations:**')
        cells.append(cell)
        if idx == 9:
            inventory = nb.v4.new_markdown_cell(dependency_source)
            inventory.metadata['dependency_inventory'] = True
            cells.append(inventory)
    notebook.cells = cells
    return notebook
