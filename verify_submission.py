#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare a generated submission with the reference submission.')
    parser.add_argument('--generated', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path('submission_reference.csv'))
    args = parser.parse_args()

    gen = pd.read_csv(args.generated)
    ref = pd.read_csv(args.reference)

    if list(gen.columns) != list(ref.columns):
        raise ValueError(f'Column mismatch: {gen.columns.tolist()} vs {ref.columns.tolist()}')
    if len(gen) != len(ref):
        raise ValueError(f'Row count mismatch: {len(gen)} vs {len(ref)}')
    if not gen['ID'].equals(ref['ID']):
        raise ValueError('ID column mismatch.')

    diff = np.abs(gen['Pred'].to_numpy() - ref['Pred'].to_numpy())
    print(f'Generated MD5: {md5(args.generated)}')
    print(f'Reference MD5: {md5(args.reference)}')
    print(f'Max abs diff: {diff.max():.12f}')
    print(f'Mean abs diff: {diff.mean():.12f}')
    print(f'Exact value match: {bool(np.array_equal(gen["Pred"].to_numpy(), ref["Pred"].to_numpy()))}')


if __name__ == '__main__':
    main()
