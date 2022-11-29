import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sslearn.model_selection import (artificial_ssl_dataset, StratifiedKFoldSS)
from sklearn.datasets import load_iris

def test_artificial_ssl_dataset():
    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, true_label = artificial_ssl_dataset(X, y, label_rate=0.1)
    assert X_unlabel.shape[0] == true_label.shape[0]
    assert X_unlabel.shape[0]/X.shape[0] == pytest.approx(0.9)

def test_StratifiedKFoldSS():
    X, y = load_iris(return_X_y=True)
    splits = 5
    skf = StratifiedKFoldSS(n_splits=splits, shuffle=False)
    for X_, y_, label, unlabel in skf.split(X, y):
        assert label.shape[0]/X_.shape[0] == pytest.approx(1/splits)
        assert unlabel.shape[0]/X_.shape[0] == pytest.approx(1-1/splits)
