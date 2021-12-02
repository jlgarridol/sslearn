from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import MetaEstimatorMixin
from sklearn.multiclass import OneVsRestClassifier, LabelBinarizer, _ConstantPredictor
from joblib import Parallel
from sklearn.utils.fixes import delayed
from sklearn.base import clone as skclone
from sklearn.utils import check_X_y


import warnings


def get_dataset(X, y):
    X, y = check_X_y(X, y)

    X_label = X[y != y.dtype.type(-1)]
    y_label = y[y != y.dtype.type(-1)]
    X_unlabel = X[y == y.dtype.type(-1)]

    return X_label, y_label, X_unlabel


class Ensemble(ABC, MetaEstimatorMixin):

    @abstractmethod
    def predict_proba(self, X, **kwards):
        pass

    def predict(self, X, **kwards):
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
        predicted_probabilitiy = self.predict_proba(X, **kwards)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)


def _fit_binary_ssl(estimator, X_label, y_label, X_unlabel, classes=None):
    unique_y = np.unique(y_label)
    X = np.concatenate((X_label, X_unlabel), axis=0)
    y = np.concatenate((y_label, np.array([y_label.dtype.type(-1)] * X_unlabel.shape[0])))
    if len(unique_y) == 1:
        if classes is not None:
            if y_label[0] == -1:
                c = 0
            else:
                c = y_label[0]
            warnings.warn(
                "Label %s is present in all training examples." % str(classes[c])
            )
        estimator = _ConstantPredictor().fit(X_label, unique_y)
    else:
        estimator = skclone(estimator)
        estimator.fit(X, y)
    return estimator


class OneVsRestSSLClassifier(OneVsRestClassifier):

    def __init__(self, estimator, *, n_jobs=None):
        super().__init__(estimator, n_jobs=n_jobs)

    def fit(self, X, y):

        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y_label)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_binary_ssl)(
                self.estimator,
                X_label,
                column,
                X_unlabel,
                classes=[
                    "not %s" % self.label_binarizer_.classes_[i],
                    self.label_binarizer_.classes_[i],
                ],
            )
            for i, column in enumerate(columns)
        )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self
