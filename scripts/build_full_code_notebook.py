"""Validate and export the executed, template-aligned final notebook.

To rebuild the analysis itself, run scripts/align_template_notebook.py.
This exporter deliberately does not overwrite the submission with the old stage assembly.
"""
from pathlib import Path
import re
import nbformat as nb
from nbconvert import HTMLExporter
from submission_narrative import clean_submission, INSTRUCTION_HEADING
ROOT = Path(__file__).resolve().parents[1]

def headings(notebook):
    return [line.strip() for cell in notebook.cells if cell.cell_type == 'markdown'
            for line in cell.source.splitlines() if re.match(r'^#{1,6} ', line)]

def main():
    path=ROOT/'notebooks/AIML_Project_1_Full_Code_Notebook_Completed.ipynb'
    notebook=nb.read(path,4)
    reference=nb.read(ROOT/'notebooks/reference/AIML_Project_1_Full_Code_Notebook.ipynb',4)
    notebook=clean_submission(notebook, reference)
    expected=[h for h in headings(reference) if h != INSTRUCTION_HEADING]
    actual=headings(notebook)
    if actual[:len(expected)] != expected:
        raise RuntimeError('Template headings or their order have changed.')
    nb.validate(notebook)
    for i,cell in enumerate(notebook.cells):
        if cell.cell_type != 'code': continue
        compile(cell.source,f'cell-{i}','exec')
        if cell.execution_count is None: raise RuntimeError(f'Cell {i} has not executed.')
        if any(o.output_type=='error' for o in cell.outputs): raise RuntimeError(f'Cell {i} contains an error.')
    nb.write(notebook,path)
    exporter=HTMLExporter(); exporter.embed_images=True
    document,_=exporter.from_notebook_node(notebook)
    (ROOT/'reports/AIML_Project_1_Full_Code_Notebook_Completed.html').write_text(document)
    site_document = document.replace('href="archive/', 'href="https://github.com/Namees-aLbayati/wind-turbine-failure-detection/blob/main/notebooks/archive/')
    for destination in [ROOT/'index.html', ROOT/'docs/index.html']:
        destination.write_text(site_document)
    print(f'Validated {len(expected)} analysis headings and exported executed notebook HTML.')
if __name__=='__main__': main()
