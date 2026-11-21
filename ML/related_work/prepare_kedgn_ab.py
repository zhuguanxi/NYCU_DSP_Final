"""Prepare all P12 A/B raw records for KEDGN, without reading test outcomes."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ML = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML / 'src'))
from prepare import DATA, OUT, sha
from kedgn_transfer_inputs import AUTHOR, raw_record


def main(folder):
    if folder.exists():
        raise FileExistsError(folder)
    labels_path, folds_path = OUT / 'labels_AB.csv', OUT / 'folds3.csv'
    labels = pd.read_csv(labels_path, index_col='RecordID').In_hospital_death
    folds = pd.read_csv(folds_path, index_col='RecordID').loc[labels.index].fold3
    variables = np.load(AUTHOR / 'data/physionet/processed_data/ts_params.npy', allow_pickle=True).tolist()
    assert len(labels) == 8000 and int(labels.sum()) == 1122
    assert labels.index.is_unique and set(folds.unique()) == {0, 1, 2}
    records, raw_hashes = {}, {}
    for part in ('a', 'b'):
        paths = sorted((DATA / f'set-{part}').glob('*.txt'))
        if len(paths) != 4000:
            raise ValueError(f'Expected 4000 Set {part} raw records')
        for path in paths:
            stay = int(path.stem)
            if stay in records:
                raise ValueError('Duplicate A/B ID')
            records[stay] = raw_record(path.read_bytes(), stay, variables)
            raw_hashes[f'set-{part}/{path.name}'] = sha(path)
        print(f'Parsed Set {part}: {len(paths)} records', flush=True)
    if set(records) != set(labels.index):
        raise ValueError('A/B raw records differ from the original development cohort')
    rows = [records[stay] for stay in labels.index]
    sizes = np.asarray([len(row['arr']) for row in rows], dtype=np.int64)
    folder.mkdir(parents=True)
    path = folder / 'ab.npz'
    np.savez_compressed(path, ids=labels.index.to_numpy(), y=labels.to_numpy(dtype=np.int64),
                        folds=folds.to_numpy(dtype=np.int64), offsets=np.r_[0, np.cumsum(sizes)],
                        arr=np.concatenate([row['arr'] for row in rows]),
                        time=np.concatenate([row['time'] for row in rows]),
                        static=np.asarray([row['extended_static'] for row in rows], dtype=np.float64),
                        lengths=np.asarray([row['length'] for row in rows], dtype=np.int64))
    manifest = {
        'stays': len(labels), 'deaths': int(labels.sum()), 'dynamic_channels': variables,
        'folds': {str(f): {'stays': int(sum(folds == f)), 'deaths': int(labels[folds == f].sum())} for f in range(3)},
        'maximum_length': max(row['length'] for row in rows),
        'zero_length_stays': sum(row['length'] == 0 for row in rows),
        'static_order_repaired': sum(row['static_order_repaired'] for row in rows),
        'files_sha256': {'ab.npz': sha(path)},
        'source_sha256': {str(p): sha(p) for p in (labels_path, folds_path,
                           AUTHOR / 'model.py', AUTHOR / 'utils.py', AUTHOR / 'train.py',
                           AUTHOR / 'data/physionet/physionet_bert_var_rep_gpt_source.pt')},
        'raw_files_sha256': raw_hashes,
        'test_labels_read': False, 'model_fits': 0,
        'parser': 'P53 raw_record; author positive-value mask; minute < 2880; static by name',
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('stays','deaths','folds','maximum_length','zero_length_stays','static_order_repaired')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT / 'related_work/KEDGN/ab_dense_gpu_20261121/inputs')
    main(parser.parse_args().output.resolve())
