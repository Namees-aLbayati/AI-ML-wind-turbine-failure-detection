"""Execute dependency-ordered notebooks, saving completed output even on failure."""
import argparse
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('notebooks', nargs='+')
    args = parser.parse_args()
    for name in args.notebooks:
        path = ROOT / 'notebooks' / name
        notebook = nbformat.read(path, 4)
        print(f'Executing {name}', flush=True)
        def progress(cell, cell_index, **kwargs):
            print(f'{name}: cell {cell_index}', flush=True)
        try:
            NotebookClient(notebook, timeout=3600, resources={'metadata': {'path': str(ROOT)}},
                           on_cell_start=progress).execute()
        finally:
            nbformat.write(notebook, path)
        from refresh_notebook_summaries import refresh
        refresh(name)
        print(f'Completed {name}', flush=True)

if __name__ == '__main__':
    main()
