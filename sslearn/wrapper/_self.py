from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.base import clone as skclone


class SelfTraining(ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(),
                 label_criteria=0.5, max_iterations=100, stop_proportion=0.0):
        """[summary]

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            estimator to be iteratively fitted , by default DecisionTreeClassifier()
        label_criteria : float, optional
            P(y|x) needed to pseudolabel x instance with y label, by default 0.5
        max_iterations : int, optional
            number of iterations of training, -1 if no max iterations, by default 100
        stop_proportion : float, optional
            proportion between unlabel and label instances to stop fitting, by default 0.0
        """
        assert isinstance(base_estimator, ClassifierMixin), "This method only support classification"
        assert stop_proportion >= 0.0 and stop_proportion < 1.0, "Stop proportion cannot be outside [0.0, 1.0)"
        assert max_iterations > 0 or max_iterations == -1, "Max iterations cannot be 0 or less than -1"

        self.base_estimator = skclone(base_estimator)
        self.label_criteria = label_criteria
        self.max_iterations = max_iterations
        self.stop_proportion = stop_proportion

    def fit(self, X, y, **kwards):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabel.

        Returns
        -------
        self: SelfTraining
            Fitted estimator.
        """    
        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        self.base_estimator.fit(X_label, y_label, **kwards)
        faults = 0  # If there are three faults then the model has converged
        self.classes_ = self.h_[0].classes_
        it = 0
        while len(X_unlabel != 0) and \
                len(X_unlabel)/len(X_label) > self.stop_proportion:
            if it == -1 or it < self.max_iterations or faults == 3:
                raw_predictions = self.base_estimator.predict_proba(X_unlabel)

                predictions = np.max(raw_predictions, axis=1)
                class_predicted = np.argmax(raw_predictions, axis=1)

                to_label = predictions >= self.label_criteria
                X_label = np.append(X_label, X_unlabel[to_label], axis=0)
                X_unlabel = X_unlabel[~to_label]
                pseudoy = np.array(list(map(lambda x: self.classes_[x], class_predicted[to_label])))
                y_label = np.append(y_label, pseudoy, axis=0)

                self.base_estimator.fit(X_label, y_label, **kwards)
                it += 1

                if len(to_label) == 0:
                    faults += 1
                else:
                    faults = 0

            else:
                break
        return self

    def predict(self, X):
        """Predict class value for X.

        For a classification model, the predicted class for each sample in X is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y: array-like of shape (n_samples,)
            The predicted classes
        """        
        check_is_fitted(self.base_estimator)
        return self.base_estimator.predict(X)

    def predict_proba(self, X, **kwards):
        """Predict class probabilities of the input samples X.

        The predicted class probability depends on the base estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y: ndarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1
            The predicted classes
        """
        check_is_fitted(self.base_estimator)
        return self.base_estimator.predict_proba(X, **kwards)

    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights., by default None

        Returns
        -------
        score: float
            Mean accuracy of self.predict(X) wrt. y.
        """
        return self.base_estimator.score(X, y, sample_weight)



