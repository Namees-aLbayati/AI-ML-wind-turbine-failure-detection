"""Comparable ANN experiments on the verified notebook-03 representation."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, fbeta_score, precision_score,
                             recall_score, f1_score, roc_auc_score)


def load_data(root):
    directory = Path(root) / 'data/processed'
    manifest = json.loads((directory / '03_ann_manifest.json').read_text())
    for name, expected in manifest.items():
        path = Path(root) / 'data/raw/wind_turbine_detection.csv' if name == 'raw_sha256' else directory / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Preprocessing artifact changed: {name}. Rerun notebook 03.')
    metadata = {s: pd.read_csv(directory / f'03_{s}_metadata.csv') for s in ['train', 'validation', 'test']}
    test_ids = set(metadata['test'].turbine_id)
    assert not test_ids.intersection(metadata['train'].turbine_id)
    assert not test_ids.intersection(metadata['validation'].turbine_id)
    with np.load(directory / '03_ann_arrays.npz') as arrays:
        data = {k: arrays[k] for k in arrays.files}
    for split, frame in metadata.items():
        assert np.array_equal(data[f'y_{split}'], frame.failure.to_numpy())
        assert len(data[f'X_{split}']) == len(frame)
        assert np.isfinite(data[f'X_{split}']).all()
    return data


def metrics(y, probability, threshold=0.5):
    pred = np.asarray(probability) >= threshold
    return dict(recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                f2=fbeta_score(y, pred, beta=2, zero_division=0),
                pr_auc=average_precision_score(y, probability),
                roc_auc=roc_auc_score(y, probability))


CONFIGS = [
    dict(name='linear_sgd', hidden=(), optimizer='sgd', lr=0.01, batch=32, dropout=0.),
    dict(name='128_64_sgd', hidden=(128, 64), optimizer='sgd', lr=0.01, batch=32, dropout=0.),
    dict(name='128_64_adam', hidden=(128, 64), optimizer='adam', lr=1e-4, batch=32, dropout=0.),
    dict(name='128_64_bn_dropout', hidden=(128, 64), optimizer='adam', lr=1e-4, batch=128, dropout=0.5),
    dict(name='64_32_adam_control', hidden=(64, 32), optimizer='adam', lr=0.001, batch=256, dropout=0.),
]


def dataset(X, y=None, batch=256, weights=None, training=False):
    import tensorflow as tf
    tensors = X if y is None else (X, y) if weights is None else (X, y, weights)
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    if training:
        ds = ds.shuffle(len(X), seed=42)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    return ds.batch(batch).with_options(options).prefetch(1)


def train_comparison(data, output, epochs=30):
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.utils.class_weight import compute_sample_weight
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = compute_sample_weight('balanced', data['y_train']).astype('float32')
    rows, histories = [], {}
    for config in CONFIGS:
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        model = keras.Sequential([keras.layers.Input((data['X_train'].shape[1],))])
        for width in config['hidden']:
            model.add(keras.layers.Dense(width, activation='relu',
                kernel_regularizer=keras.regularizers.l2(1e-4)))
            if config['dropout']:
                model.add(keras.layers.BatchNormalization())
                model.add(keras.layers.Dropout(config['dropout']))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        optimizer = (keras.optimizers.SGD(learning_rate=config['lr'], momentum=0.9)
                     if config['optimizer']=='sgd' else keras.optimizers.Adam(learning_rate=config['lr']))
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
        history = model.fit(
            dataset(data['X_train'], data['y_train'], config['batch'], weights, True),
            validation_data=dataset(data['X_validation'], data['y_validation']),
            epochs=epochs, verbose=2,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                       keras.callbacks.TerminateOnNaN()])
        if not all(np.isfinite(v).all() for v in history.history.values()):
            raise RuntimeError(f"Nonfinite training history: {config['name']}")
        histories[config['name']] = history.history
        model.save(output / f"{config['name']}.keras")
        for split in ['train', 'validation']:
            probability = model.predict(dataset(data[f'X_{split}']), verbose=0).ravel()
            rows.append(dict(model=config['name'], split=split, epochs=len(history.history['loss']),
                             **metrics(data[f'y_{split}'], probability)))
            if split == 'validation':
                np.save(output / f"{config['name']}_validation_probability.npy", probability)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / 'comparison.csv', index=False)
    (output / 'histories.json').write_text(json.dumps(histories, indent=2))
    (output / 'configs.json').write_text(json.dumps(CONFIGS, indent=2))
    # Model ranking uses PR-AUC; the operating threshold separately maximizes F2.
    winner = comparison.query("split == 'validation'").sort_values(['pr_auc', 'f2'], ascending=False).iloc[0]['model']
    probability = np.load(output / f'{winner}_validation_probability.npy')
    thresholds = pd.DataFrame([dict(threshold=float(t), **metrics(data['y_validation'], probability, t))
                              for t in np.linspace(0.01, 0.99, 197)])
    thresholds.to_csv(output / 'validation_thresholds.csv', index=False)
    best = thresholds.sort_values(['f2', 'precision', 'threshold'], ascending=False).iloc[0]
    selection = dict(model=winner, threshold=float(best.threshold), criterion='validation PR-AUC; threshold by validation F2')
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    return comparison, histories, selection


def evaluate_selected(data, output):
    from tensorflow import keras
    output = Path(output)
    selection = json.loads((output / 'selection.json').read_text())
    model = keras.models.load_model(output / f"{selection['model']}.keras")
    probability = model.predict(dataset(data['X_test']), verbose=0).ravel()
    result = dict(**selection, **metrics(data['y_test'], probability, selection['threshold']))
    (output / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    np.save(output / 'test_probability.npy', probability)
    return result, probability
