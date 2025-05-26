import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.dummy import DummyClassifier, DummyRegressor

from sslearn.base import get_dataset, get_dataset_regression
from sslearn.datasets import read_csv, read_keel, save_keel, secure_dataset
from sslearn.wrapper import SelfTraining, TriTrainingRegressor


def folder():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "example_files")

def posterior(X, y):
    X_label, y_label, X_unlabel = get_dataset(X, y)
    assert X_unlabel.shape[0] != 0
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_label, y_label)
    clf = SelfTraining(DummyClassifier(strategy="most_frequent"))
    clf.fit(X, y)

def posterior_regression(X, y): 
    X_label, y_label, X_unlabel = get_dataset_regression(X, y)
    assert X_unlabel.shape[0] != 0
    reg = DummyRegressor(strategy="mean")
    reg.fit(X_label, y_label)
    reg = TriTrainingRegressor(DummyRegressor(strategy="mean"))
    reg.fit(X, y)


class TestDataset:
    
    def test_read_csv(self):
        X, y = read_csv(os.path.join(folder(),"abalone.csv"), format="pandas")
        posterior(X, y)
        X, y = read_csv(os.path.join(folder(),"abalone.csv"), format="numpy")
        posterior(X, y)

    def test_read_csv_regression(self): 
        X, y = read_csv(os.path.join(folder(), "abalone_regression.csv"), format="pandas", is_regression=True)
        posterior_regression(X, y)
        X, y = read_csv(os.path.join(folder(), "abalone_regression.csv"), format="numpy", is_regression=True)
        posterior_regression(X, y)


    def test_read_keel(self):
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="pandas")
        posterior(X, y)
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="numpy")
        posterior(X, y)

    def test_read_keel_regression(self):
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="pandas", is_regression=True)
        posterior_regression(X, y)
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="numpy", is_regression=True)
        posterior_regression(X, y)

    def test_secure_dataset(self):
        X, y = read_csv(os.path.join(folder(),"abalone.csv"), format="pandas")
        X_label, y_label, _ = get_dataset(X, y)
        X1, y1 = secure_dataset(X_label, y_label)
        with pytest.raises(ValueError):
            secure_dataset(X, y)
        assert (X1.values == X_label.values).all()
        assert (y1 == y_label).all()

    def test_save_keel(self):
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="pandas")
        save_keel(X, y, os.path.join(folder(),"temp_abalone.dat"), name="abalone")
        X1, y1 = read_keel(os.path.join(folder(),"temp_abalone.dat"), format="pandas")
        assert (X.columns == X1.columns).all()
        assert (y == y1).all()
        assert (X == X1).all().all()
