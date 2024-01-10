import array
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.base import clone as skclone
from sklearn.base import is_classifier
from sklearn.multiclass import (LabelBinarizer, OneVsRestClassifier,
                                _ConstantPredictor, _num_samples,
                                _predict_binary)
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.metaestimators import available_if
from sklearn.ensemble._base import _set_random_states
from sklearn.utils import check_random_state

def get_dataset(X, y):

    is_df = False
    if isinstance(X, pd.DataFrame):
        is_df = True
        columns = X.columns

    X = check_array(X)
    y = check_array(y, ensure_2d=False, dtype=y.dtype.type)
    
    X_label = X[y != y.dtype.type(-1)]
    y_label = y[y != y.dtype.type(-1)]
    X_unlabel = X[y == y.dtype.type(-1)]

    X_label, y_label = check_X_y(X_label, y_label)

    if is_df:
        X_label = pd.DataFrame(X_label, columns=columns)
        X_unlabel = pd.DataFrame(X_unlabel, columns=columns)

    return X_label, y_label, X_unlabel


class BaseEnsemble(ABC, MetaEstimatorMixin):

    @abstractmethod
    def predict_proba(self, X):
        pass

    def predict(self, X):
        """Predict the classes of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array with predicted labels.
        """
        predicted_probabilitiy = self.predict_proba(X)
        classes = self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

        # If exists label_encoder_ attribute, use it to transform classes
        if hasattr(self, "label_encoder_"):
            classes = self.label_encoder_.inverse_transform(classes)
            
        return classes


class FakedProbaClassifier(MetaEstimatorMixin, ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.one_hot = OneHotEncoder().fit(y.reshape(-1, 1))
        self.base_estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        if "predict_proba" in dir(self.base_estimator):
            return self.base_estimator.predict_proba(X)
        else:
            return self.one_hot.transform(self.base_estimator.predict(X).reshape(-1, 1)).toarray()


def _fit_binary_ssl(estimator, X, y_label, size, classes=None, **fit_params):
    # unique_y = np.unique(y_label)
    # X = np.concatenate((X_label, X_unlabel), axis=0)
    y = np.concatenate((y_label, np.array([y_label.dtype.type(-1)] * size)))
    unique_y = np.unique(y_label)
    if len(unique_y) == 1:
        if classes is not None:
            if y_label[0] == -1:
                c = 0
            else:
                c = y_label[0]
            warnings.warn(
                "Label %s is present in all training examples." % str(classes[c])
            )
        estimator = _ConstantPredictor().fit(None, unique_y)
    else:
        estimator = skclone(estimator)
        estimator.fit(X, y, **fit_params)
    return estimator

def _predict_binary_ssl(estimator, X, **predict_params):
    """Make predictions using a single binary estimator."""
    try:
        score = np.ravel(estimator.decision_function(X, **predict_params))
    except (AttributeError, NotImplementedError):
        # probabilities of the positive class
        score = estimator.predict_proba(X, **predict_params)[:, 1]
    return score


class OneVsRestSSLClassifier(OneVsRestClassifier):

    def __init__(self, estimator, *, n_jobs=None):
        """Adapted OneVsRestClassifier for SSL datasets

        Parameters
        ----------
        estimator : {ClassifierMixin, list},
            An estimator object implementing fit and predict_proba or a list of ClassifierMixin
        n_jobs : n_jobs : int, optional
            The number of jobs to run in parallel. -1 means using all processors., by default None
        """
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y, **fit_params):
        #
        y_label = y[y != y.dtype.type(-1)]
        size = len(y) - len(y_label)

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y_label)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        estimators = [skclone(self.estimator) for _ in range(len(self.classes_))]
        rs = check_random_state(estimators[0].get_params(deep=False).get("random_state", None))
        for e in estimators:
            _set_random_states(e, rs)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary_ssl)(
                estimators[i],
                X,
                column,
                size,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
                **fit_params
            )
            for i, column in enumerate(columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

    def predict(self, X, **kwards):
        check_is_fitted(self)

        n_samples = _num_samples(X)
        if self.label_binarizer_.y_type_ == "multiclass":
            maxima = np.empty(n_samples, dtype=float)
            maxima.fill(-np.inf)
            argmaxima = np.zeros(n_samples, dtype=int)
            for i, e in enumerate(self.estimators_):
                pred = _predict_binary_ssl(e, X, **kwards)
                np.maximum(maxima, pred, out=maxima)
                argmaxima[maxima == pred] = i
            return self.classes_[argmaxima]
        else:
            if (hasattr(self.estimators_[0], "decision_function") and
                    is_classifier(self.estimators_[0])):
                thresh = 0
            else:
                thresh = .5
            indices = array.array('i')
            indptr = array.array('i', [0])
            for e in self.estimators_:
                indices.extend(np.where(_predict_binary_ssl(e, X, **kwards) > thresh)[0])
                indptr.append(len(indices))
            data = np.ones(len(indices), dtype=int)
            indicator = sp.csc_matrix((data, indices, indptr),
                                      shape=(n_samples, len(self.estimators_)))
            return self.label_binarizer_.inverse_transform(indicator)

    def predict_proba(self, X, **kwards):
        check_is_fitted(self)
        # Y[i, j] gives the probability that sample i has the label j.
        # In the multi-label case, these are not disjoint.
        Y = np.array([e.predict_proba(X, **kwards)[:, 1] for e in self.estimators_]).T

        if len(self.estimators_) == 1:
            # Only one estimator, but we still want to return probabilities
            # for two classes.
            Y = np.concatenate(((1 - Y), Y), axis=1)

        if not self.multilabel_:
            # Then, probabilities should be normalized to 1.
            Y /= np.sum(Y, axis=1)[:, np.newaxis]
        return Y