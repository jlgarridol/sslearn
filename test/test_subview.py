import os
import sys
import pandas as pd
import numpy as np

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sslearn.subview import SubViewClassifier, SubViewRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_diabetes

def test_subview_exceptions():
    """Test that exceptions are raised when expected."""
    X, y = load_iris(return_X_y=True)
    clf = SubViewClassifier(DecisionTreeClassifier(), subview="a")
    with pytest.raises(ValueError):
        clf.fit(X, y)
    with pytest.raises(ValueError):
        clf = SubViewClassifier(DecisionTreeClassifier(), subview="a", mode="invalid")
    
class TestSubViewClassifier:
    
    def test_SubViewClassifier_regex(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X)
        X.columns = ["a", "b", "c", "d"]
        clf = SubViewClassifier(DecisionTreeClassifier(random_state=0), mode="regex", subview="a|b")
        clf2 = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)
        clf2.fit(X[["a", "b"]], y)
        assert (clf.predict(X) == clf2.predict(X[["a", "b"]])).all()

    def test_SubViewClassifier_include(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X)
        X.columns = ["a1", "a2", "b1", "b2"]
        clf = SubViewClassifier(DecisionTreeClassifier(random_state=0), mode="include", subview="a")
        clf2 = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)
        clf2.fit(X[["a1", "a2"]], y)
        assert (clf.predict(X) == clf2.predict(X[["a1", "a2"]])).all()

    def test_SubViewClassifier_index(self):
        X, y = load_iris(return_X_y=True)
        clf = SubViewClassifier(DecisionTreeClassifier(random_state=0), mode="index", subview=[0, 1])
        clf2 = DecisionTreeClassifier(random_state=0)
        clf.fit(X, y)
        clf2.fit(X[:, [0, 1]], y)
        assert (clf.predict(X) == clf2.predict(X[:, [0, 1]])).all()

class TestSubViewRegressor:

    def test_SubViewRegressor_regex(self):
        X, y = load_diabetes(return_X_y=True)
        X = pd.DataFrame(X)
        X.columns = ["a"+str(i) for i in range(X.shape[1])]
        clf = SubViewRegressor(DecisionTreeRegressor(random_state=0), mode="regex", subview="[1-3]")
        clf2 = DecisionTreeRegressor(random_state=0)
        clf.fit(X, y)
        clf2.fit(X[["a1", "a2", "a3"]], y)
        np.testing.assert_allclose(clf.predict(X)[::-20], clf2.predict(X[["a1", "a2", "a3"]])[::-20])
    