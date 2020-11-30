from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score


class SelfTraining(ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(), label_criteria=0.5,
                 max_iterations=100, stop_proportion=0.0):
        self.base_estimator = base_estimator
        self.label_criteria = label_criteria
        self.max_iterations = max_iterations
        self.stop_proportion = stop_proportion

    def fit(self, X, y):
        X_label = X[y != -1]
        y_label = y[y != -1]
        X_unlabel = X[y == -1]

        self.base_estimator.fit(X_label, y_label)
        classes = self.base_estimator.classes_
        it = 0
        while len(X_unlabel != 0) and \
                len(X_unlabel)/len(X_label) > self.stop_proportion:
            if it == -1 or it < self.max_iterations:

                predictions = np.argmax(self.base_estimator.predict_proba(X_unlabel), axis=1)
                to_label = predictions >= self.label_criteria
                X_label = np.append(X_label, X_unlabel[to_label], axis=0)
                X_unlabel = X_unlabel[~to_label]
                pseudoy = np.array(list(map(lambda x: classes[x], predictions[to_label])))
                y_label = np.append(y_label, pseudoy, axis=0)
                
                self.base_estimator.fit(X_label, y_label)
                it += 1
            else:
                break
        return self

    def predict(self, X):
        check_is_fitted(self.base_estimator)
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self.base_estimator)
        return self.base_estimator.predict_proba(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y, sample_weight=None):
        y_test = self.base_estimator.predict(X)
        return accuracy_score(y, y_test)
