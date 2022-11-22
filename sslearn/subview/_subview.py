import numpy as np
import pandas as pd
from sklearn.utils import check_X_y
from sklearn.base import clone as skclone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, is_classifier, MetaEstimatorMixin
import re


class SubView(BaseEstimator):

    def __init__(self, base_estimator, subview, mode="regex"):
        valid_modes = ["regex", "index", "include"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        self.base_estimator = base_estimator
        self.subiew = subview
        self.mode = mode
        if mode == "regex" or mode == "include":
            if type(subview) != str:
                raise ValueError(f"Subview must be a string when mode is {mode}")

    def fit(self, X, y, **kwards):
        X, y = check_X_y(X, y)
        self.base_estimator_ = skclone(self.base_estimator)

        if self.mode == "regex":
            self.subview_ = re.compile(self.subiew)
            X = self._regex_subview(X)
        elif self.mode == "index":
            X = self._index_subview(X)
        elif self.mode == "include":
            X = self._include_subview(X)

        self.base_estimator_.fit(X, y)

        if is_
        self.classes_ = np.unique(y)

    def predict(self, X):
        if self.mode == "regex":
            X = self._regex_subview(X)
        elif self.mode == "index":
            X = self._index_subview(X)
        elif self.mode == "include":
            X = self._include_subview(X)

        return self.base_estimator_.predict(X)

    def _regex_subview(self, X):
        if isinstance(X, pd.DataFrame):
            return X.filter(regex=self.subview_)
        else:
            raise ValueError("Subview mode 'regex' and 'include' only works with pandas DataFrames")

    def _include_subview(self, X):
        return self._regex_subview(X)

    def _index_subview(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.subview_]
        else:
            return X[:, self.subview_]


class SubViewClassifier(SubView, ClassifierMixin):

    def predict_proba(self, X):
        if self.mode == "regex":
            X = self._regex_subview(X)
        elif self.mode == "index":
            X = self._index_subview(X)
        elif self.mode == "include":
            X = self._include_subview(X)

        return self.base_estimator_.predict_proba(X)

class SubViewRegressor(SubView, RegressorMixin):

    def predict(self, X):
        return super.predict(X)