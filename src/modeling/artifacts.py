"""Artifact integrity and model adapters shared by downstream notebooks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

class BalancedXGBClassifier(XGBClassifier):
    """Recompute class weighting within each fit, including every CV fold."""
    def fit(self, X, y, **kwargs):
        labels = np.asarray(y)
        self.set_params(scale_pos_weight=float((labels == 0).sum() / (labels == 1).sum()))
        return super().fit(X, y, **kwargs)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_splits(root):
    root = Path(root)
    from src.modeling.neural_networks import load_data
    load_data(root)
    metadata = {s: pd.read_csv(root / f'data/processed/03_{s}_metadata.csv', parse_dates=['timestamp'])
                for s in ['train', 'validation', 'test']}
    frames = {s: pd.read_csv(root / f'data/processed/04_{s}_engineered.csv', parse_dates=['timestamp'])
              for s in metadata}
    for split, frame in frames.items():
        assert frame.source_index.is_unique
        left = frame.set_index('source_index')[['turbine_id', 'timestamp', 'failure']].sort_index()
        right = metadata[split].set_index('source_index')[['turbine_id', 'timestamp', 'failure']].sort_index()
        pd.testing.assert_frame_equal(left, right, check_dtype=False)
    test_ids = set(metadata['test'].turbine_id)
    assert not test_ids.intersection(metadata['train'].turbine_id)
    assert not test_ids.intersection(metadata['validation'].turbine_id)
    for turbine, train in frames['train'].groupby('turbine_id'):
        assert train.timestamp.max() < frames['validation'].query('turbine_id == @turbine').timestamp.min()
    return frames

def write_manifest(root, paths, destination):
    root = Path(root)
    record = {str(Path(p).relative_to(root)): digest(p) for p in paths}
    Path(destination).write_text(json.dumps(record, indent=2))

def verify_manifest(root, path):
    root = Path(root)
    for relative, expected in json.loads(Path(path).read_text()).items():
        if digest(root / relative) != expected:
            raise ValueError(f'Artifact changed: {relative}. Rerun its producer and downstream stages.')
