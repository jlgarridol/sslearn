import re

import numpy as np
import pandas as pd
from sklearn.base import (BaseEstimator, ClassifierMixin, MetaEstimatorMixin,
                          RegressorMixin)
from sklearn.base import clone as skclone
from sklearn.base import is_classifier
from sklearn.utils import check_X_y


class SubView(BaseEstimator):

    def __init__(self, base_estimator, subview, mode="regex"):
        """Create a classifier that uses a subview of the data.

        Parameters
        ----------
        base_estimator : BaseEstimator
            The base estimator to use for fitting and predicting.
        subview : str or list
            The subview to use for fitting and predicting.
        mode : str, optional
            The mode to use for the subview. Can be 'regex', 'index' or 'include'. Default is 'regex'.

        Raises
        ------
        ValueError
            If the mode is not 'regex', 'index' or 'include'.
        ValueError
            If the mode is 'regex' or 'include' and the subview is not a string.
        """        
        valid_modes = ["regex", "index", "include"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
        self.base_estimator = base_estimator
        self.subview = subview
        self.mode = mode
        if mode == "regex" or mode == "include":
            if type(subview) != str:
                raise ValueError(f"Subview must be a string when mode is {mode}")

    def fit(self, X, y, **kwards):
        """Fit the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels in classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If the mode is 'regex' or 'include' and the X is not a pandas DataFrame.
        """        
        df = False
        if isinstance(X, pd.DataFrame):
            df = True
            columns = X.columns
        X, y = check_X_y(X, y)
        if df:
            X = pd.DataFrame(X)
            X.columns = columns


        self.base_estimator_ = skclone(self.base_estimator)

        if self.mode == "regex" or self.mode == "include":
            self.subview = re.compile(self.subview)

        if self.mode == "regex":            
            X = self._regex_subview(X)
        elif self.mode == "index":
            X = self._index_subview(X)
        elif self.mode == "include":
            X = self._include_subview(X)

        self.base_estimator_.fit(X, y, **kwards)

        if is_classifier:
            self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """Predict using the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted values.
        """        
        if self.mode == "regex":
            X = self._regex_subview(X)
        elif self.mode == "index":
            X = self._index_subview(X)
        elif self.mode == "include":
            X = self._include_subview(X)

        return self.base_estimator_.predict(X)

    def _regex_subview(self, X):
        if isinstance(X, pd.DataFrame):
            return X.filter(regex=self.subview)
        else:
            raise ValueError("Subview mode 'regex' and 'include' only works with pandas DataFrames")

    def _include_subview(self, X):
        return self._regex_subview(X)

    def _index_subview(self, X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, self.subview]
        else:
            return X[:, self.subview]


class SubViewClassifier(SubView, ClassifierMixin):

    def predict_proba(self, X):
        """Predict class probabilities using the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """        
        if self.mode == "regex":
            X = self._regex_subview(X)
        elif self.mode == "index":
            X = self._index_subview(X)
        elif self.mode == "include":
            X = self._include_subview(X)

        return self.base_estimator_.predict_proba(X)

class SubViewRegressor(SubView, RegressorMixin):

    def predict(self, X):
        """Predict using the base estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted values.
        """        
        return super().predict(X)