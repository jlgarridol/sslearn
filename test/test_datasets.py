import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sslearn.datasets import (read_csv, read_keel, secure_dataset, save_keel)
from sslearn.base import get_dataset
from sklearn.dummy import DummyClassifier
from sslearn.wrapper import SelfTraining

def folder():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "example_files")

def posterior(X, y):
    X_label, y_label, X_unlabel = get_dataset(X, y)
    assert X_unlabel.shape[0] != 0
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(X_label, y_label)
    clf = SelfTraining(DummyClassifier(strategy="most_frequent"))
    clf.fit(X, y)

class TestDataset:
    
    def test_read_csv(self):
        X, y = read_csv(os.path.join(folder(),"abalone.csv"), format="pandas")
        posterior(X, y)
        X, y = read_csv(os.path.join(folder(),"abalone.csv"), format="numpy")
        posterior(X, y)


    def test_read_keel(self):
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="pandas")
        posterior(X, y)
        X, y = read_keel(os.path.join(folder(),"abalone.dat"), format="numpy")
        posterior(X, y)