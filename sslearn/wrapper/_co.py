from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score
from sklearn.base import clone as skclone


class CoTrainingBySubspace(ClassifierMixin):

    def __init__(self, base_classifier=DecisionTreeClassifier(), max_iterations=100, label_criteria=0.5, random_state=None):
        assert isinstance(base_classifier, ClassifierMixin), "This method only support classification"
        self.base_classifier = base_classifier
        self.max_iterations = max_iterations
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        X_label = X[y != -1]
        y_label = y[y != -1]
        X_unlabel = X[y == -1]

        p = self.random_state.permutation(X_label.shape[1])
        index1 = p[0:int(len(p/2))]
        index2 = p[int(len(p)/2):]
        sub1 = X_label[:, index1]
        sub2 = X_label[:, index2]

        sub1_u = X_unlabel[:, index1]
        sub2_u = X_unlabel[:, index2]

        c1 = skclone(self.base_classifier).fit(sub1, y_label)
        c2 = skclone(self.base_classifier).fit(sub2, y_label)

        self.classes = c1.classes_

        for i in range(self.max_iterations):
            if len(sub1_u) == 0 or len(sub2_u) == 0:
                break
            
            

    def _prediction_process(self, classifier, space):
        raw_predictions = classifier.predict_proba(space)
        predictions = np.max(raw_predictions, axis=1)
        class_predicted = np.argmax(raw_predictions, axis=1)

        to_label = predictions >= self.label_criteria
        return to_label

class CoTrainingByCommittee(ClassifierMixin):

    def __init__(self, ensemble_estimator=BaggingClassifier(), max_iterations=100, poolsize=100, random_state=None):
        assert isinstance(ensemble_estimator, ClassifierMixin), "This method only support classification"
        self.ensemble_estimator = skclone(ensemble_estimator)
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        X_label = X[y != -1]
        y_label = y[y != -1]
        X_unlabel = X[y == -1]
        prior = self._calculate_prior(y_label)
        permutation = self.random_state.permutation(len(X_unlabel))

        self.ensemble_estimator.fit(X_label, y_label)
        classes = self.ensemble_estimator.classes_

        for _ in range(self.max_iterations):
            if len(permutation) == 0:
                break
            raw_predictions = self.ensemble_estimator.predict_proba(X_unlabel[permutation[0:self.poolsize]])

            predictions = np.max(raw_predictions, axis=1)
            class_predicted = np.argmax(raw_predictions, axis=1)

            to_label = None
            for c in range(len(classes)):
                to_add = np.logical_and(class_predicted == c, predictions >= prior[classes[c]])
                if to_label is not None:
                    to_label = np.logical_or(to_label, to_add)
                else:
                    to_label = to_add

            index = permutation[0:self.poolsize][to_label]
            X_label = np.append(X_label, X_unlabel[index], axis=0)
            pseudoy = np.array(list(map(lambda x: classes[x], class_predicted[to_label])))
            y_label = np.append(y_label, pseudoy, axis=0)
            permutation = permutation[list(map(lambda x: x not in index, permutation))]

            self.ensemble_estimator.fit(X_label, y_label)

        return self

    def _calculate_prior(self, y):
        unique, counts = np.unique(y, return_counts=True)
        u_c = dict(zip(unique, counts))
        instances = len(y)
        for u in u_c:
            u_c[u] = float(u_c[u]/instances)
        return u_c

    def predict(self, X):
        check_is_fitted(self.ensemble_estimator)
        return self.ensemble_estimator.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self.ensemble_estimator)
        return self.ensemble_estimator.predict_proba(X)

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    def score(self, X, y, sample_weight=None):
        y_test = self.ensemble_estimator.predict(X)
        return accuracy_score(y, y_test)
