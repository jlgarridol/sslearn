import os
import sys
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sslearn.model_selection import (artificial_ssl_dataset, StratifiedKFoldSS)
from sklearn.datasets import load_iris

def test_artificial_ssl_dataset():
    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, true_label = artificial_ssl_dataset(X, y, label_rate=0.1)
    assert X_unlabel.shape[0] == true_label.shape[0]
    assert X_unlabel.shape[0]/X.shape[0] == pytest.approx(0.9)

def test_artificial_ssl_dataset_with_force_minimum():
    X, y = load_iris(return_X_y=True)
    # The first class only 10 instances
    first_class = np.unique(y)[0]
    X_0 = X[y == first_class][0]
    y_0 = y[y == first_class][0]
    # Keep only 1 instance of first class
    X = X[y != first_class]
    y = y[y != first_class]
    X = np.concatenate((X, [X_0]), axis=0)
    y = np.concatenate((y, [y_0]), axis=0)    

    X, y, X_unlabel, true_label = artificial_ssl_dataset(X, y, label_rate=0.02, force_minimum=1)
    assert X_unlabel.shape[0] == true_label.shape[0]
    for i in np.unique(y):
        assert np.sum(y == i) >= 1

    pytest.raises(ValueError, artificial_ssl_dataset, X, y, label_rate=0.02, force_minimum=2)

def test_artificial_ssl_dataset_with_indexes():
    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, true_label, label, unlabel = artificial_ssl_dataset(X, y, label_rate=0.1, indexes=True)
    
    assert X_unlabel.shape[0] == unlabel.shape[0]

    try:
        X, y, X_unlabel, true_label, label, unlabel = artificial_ssl_dataset(X, y, label_rate=0.1, indexes=False)
    except ValueError:
        pass
    except:
        assert False, "Should raise ValueError if indexes=False and unpack the label and unlabel indexes."

def test_StratifiedKFoldSS():
    X, y = load_iris(return_X_y=True)
    splits = 5
    skf = StratifiedKFoldSS(n_splits=splits, shuffle=False)
    for X_, y_, label, unlabel in skf.split(X, y):
        assert label.shape[0]/X_.shape[0] == pytest.approx(1/splits)
        assert unlabel.shape[0]/X_.shape[0] == pytest.approx(1-1/splits)
