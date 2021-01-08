from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state, resample
from sklearn.exceptions import NotFittedError
from sklearn.base import clone as skclone
import math
from abc import ABC, abstractmethod
from sklearn.base import MetaEstimatorMixin


def _generate_random_subspaces(X, size=2, random_state=None):
    p = check_random_state(random_state).permutation(X.shape[1])
    idxs = []
    for i in range(size):
        _from = i*int(len(p/size))
        _to = (i+1)*int(len(p/size))
        idxs.append(p[_from:_to])
    return idxs


class _BaseCoTraining(ABC, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X, y, **kwards):
        pass

    def predict(self, X, **kwards):
        predicted_probabilitiy = self.predict_proba(Xs, **kwards)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X, **kwards):
        if "h_" in dir(self):
            ys = []
            for i in range(len(self.h_):
                ys.append(self.h_[i].predict_proba(X[:,self.columns_[i]]), **kwards)
            y = (sum(ys)/len(ys))
            return y
        else:
            raise NotFittedError("Classifier not fitted")


class DemocraticCoLearning(_BaseCoTraining):

    def __init__(self, estimators=[DecisionTreeClassifier()], random_state=None):
        self.estimators = estimators
        self.random_state = check_random_state(random_state)

    def fit(self, X, y, **kwards):
        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        L = [X_label] * len(self.estimators)
        Ly = [y_label] * len(self.estimators)
        e = [0] * len(self.estimators)

        changed = True
        while changed:
            changed = False
            for i in range(len(self.estimators)):
                self.estimators[i].fit(L[i], Ly[i], **kwards)
            
            y_preds = []
            for i in range(len(self.estimators)):
                y_preds.append(self.estimators[i].predict(X_unlabel))

class CoTraining(_BaseCoTraining):
    """
    Implementation based on https://github.com/jjrob13/sklearn_cotraining

    Avrim Blum and Tom Mitchell. 1998. 
    Combining labeled and unlabeled data with co-training. 
    In Proceedings of the eleventh annual conference on Computational learning theory (COLT' 98). 
    Association for Computing Machinery, New York, NY, USA, 92–100. 
    DOI:https://doi.org/10.1145/279943.279962
    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), second_base_estimator=None,
                 max_iterations=30, poolsize=75, positives=-1, negatives=-1,
                 random_state=None):
        """Create a CoTraining classifier

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            The classifier that will be used in the cotraining algorithm on the feature set, by default DecisionTreeClassifier()
        second_base_estimator : ClassifierMixin, optional
            The classifier that will be used in the cotraining algorithm on another feature set, if none are a clone of base_estimator, by default None
        max_iterations : int, optional
            The number of iterations, by default 30
        poolsize : int, optional
            The size of the pool of unlabeled samples from which the classifier can choose, by default 75
        positives : int, optional
            The number of positive examples that will be 'labeled' by each classifier during each iteration
            The default is the is determined by the smallest integer ratio of positive to negative samples in L, by default -1
        negatives : int, optional
            The number of negative examples that will be 'labeled' by each classifier during each iteration
            The default is the is determined by the smallest integer ratio of positive to negative samples in L, by default -1
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None

        Raises
        ------
        ValueError
            Current implementation supports either both positives and negatives being specified, or neither
        """
        self.h = [base_estimator]
        if second_base_estimator is not None:
            self.h.append(second_base_estimator)
        else:
            self.h.append(skclone(base_estimator))
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.random_state = check_random_state(random_state)
        
        if (positives == -1 and negatives != -1) or (positives != -1 and negatives == -1):
            raise ValueError('Current implementation supports either both positives and negatives being specified, or neither')

        self.positives = positives
        self.negatives = negatives

    def fit(self, X, y, X2=None, features=None: list, **kwards):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        y : [type]
            [description]
        X2 : [type], optional
            [description], by default None
        features : [type], optional
            [description], by default None:list

        Returns
        -------
        [type]
            [description]
        """
        X = np.asarray(X)
        y = np.asarray(y)
        assert X2 is not None and features is not None, "The list of features and x2 cannot be defined at the same time"
        X1 = X
        if X2 is None and features is None:
            X2 = X.copy()
            self.columns_ = [list(range(X.shape[1]))]*2
        elif X2 is not None:
            X2 = np.asarray(X2)
        elif features is not None:
            X1 = X[:,features[0]]
            X2 = X[:,features[1]]
            self.columns_ = features

        if self.positives == -1 and self.negatives == -1:

            num_pos = sum(1 for y_i in y if y_i == 1)
            num_neg = sum(1 for y_i in y if y_i == 0)

            n_p_ratio = num_neg / float(num_pos)

            if n_p_ratio > 1:
                self.positives = 1
                self.negatives = round(self.positives*n_p_ratio)

            else:
                self.negatives = 1
                self.positives = round(self.negatives/n_p_ratio)

        assert(self.positives > 0 and self.negatives > 0 and \
               self.max_iterations > 0 and self.poolsize > 0), \
                "Parameters are inconsistent"

        # Set of unlabeled samples
        U = [i for i, y_i in enumerate(y) if y_i == -1]
        self.random_state.shuffle(U)

        U_ = U[-min(len(U), self.u_):]
        #remove the samples in U_ from U
        U = U[:-len(U_)]

        L = [i for i, y_i in enumerate(y) if y_i != -1]

        it = 0
        while it != self.max_iterations and U:
            it += 1

            self.h[0].fit(X1[L], y[L], **kwards)
            self.h[1].fit(X2[L], y[L], **kwards)

            if len(self.h[0].classes_):
                raise Exception("CoTraining does not support multiclass")

            y1_prob = self.h[0].predict_proba(X1[U_])
            y2_prob = self.h[1].predict_proba(X2[U_])

            n, p = [], []

            for i in (y1_prob[:,0].argsort())[-self.negatives:]:
                if y1_prob[i,0] > 0.5:
                    n.append(i)
            for i in (y1_prob[:,1].argsort())[-self.positives:]:
                if y1_prob[i,1] > 0.5:
                    p.append(i)

            for i in (y2_prob[:,0].argsort())[-self.negatives:]:
                if y2_prob[i,0] > 0.5:
                    n.append(i)
            for i in (y2_prob[:,1].argsort())[-self.positives:]:
                if y2_prob[i,1] > 0.5:
                    p.append(i)

            y[[U_[x] for x in p]] = 1
            y[[U_[x] for x in n]] = 0

            L.extend([U_[x] for x in p])
            L.extend([U_[x] for x in n])

            U_ = [elem for elem in U_ if not (elem in p or elem in n)]

            add_counter = 0 #number we have added from U to U_
            num_to_add = len(p) + len(n)
            while add_counter != num_to_add and U:
                add_counter += 1
                U_.append(U.pop())

        self.h[0].fit(X1[L], y[L], **kwards)
        self.h[1].fit(X2[L], y[L], **kwards)
        self.h_ = self.h
        self.classes_ = self.h_[0].classes_


        return self

    def predict_proba(self, X, X2=None, **kwards):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        X2 : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        NotFittedError
            [description]
        """        
        if "columns_" in self:
            return super().predict_proba(X, **kwards)
        else:
            if "h_" in dir(self):
                ys = []
                ys.append(self.h_[0].predict_proba(X, **kwards))
                ys.append(self.h_[1].predict_proba(X2, **kwards))
                y = (sum(ys)/len(ys))
                return y
            else:
                raise NotFittedError("Classifier not fitted")

    def predict(self, X, X2=None, **kwards):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        X2 : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """        
        if "columns_" in self:
            return super().predict(X, **kwards)
        else:
            predicted_probabilitiy = self.predict_proba(X, X2, **kwards)
            return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                            axis=0)



class Rasco(_BaseCoTraining):  # No estoy seguro que esto sea Rasco
    """

    """

    def __init__(self, base_estimator=DecisionTreeClassifier(),
                 max_iterations=100, label_criteria=0.5, random_state=None):
        assert isinstance(base_estimator, ClassifierMixin), "This method only support classification"
        self.base_estimator = base_estimator
        self.max_iterations = max_iterations
        self.random_state = check_random_state(random_state)
        self._n_of_subpaces = 2

    def fit(self, X, y, **kwards):
        ys = []
        for i in range(self._n_of_subpaces):
            ys.append(y)

        idxs = _generate_random_subspaces(X, self._n_of_subspaces, self.random_state)

        cfs = []
        for i in range(len(ys)):
            cfs.append(skclone(self.base_estimator)
                       .fit(X[ys[i] != y.dtype.type(-1),
                            idxs[i]], ys[i][ys[i] != y.dtype.type(-1)],
                            **kwards))

        anothery = ys.copy()  # The prediction is over the unlabel set of the another estimator
        _y = anothery.pop(0)
        anothery.append(_y)

        for _ in range(self.max_iterations):
            exit = False
            for i in ys:
                if y.dtype.type(-1) not in i:
                    exit = True
                    break
            if exit:
                break

            for i in range(self._n_of_subpaces):
                to_predict = anothery[i] == y.dtype.type(-1)

                raw_predictions = cfs[i].predict_proba(X[to_predict, idxs[i]])
                predictions = np.max(raw_predictions, axis=1)
                class_predicted = np.argmax(raw_predictions, axis=1)

                to_label = to_predict[to_predict] == (predictions >= self.label_criteria)

                anothery[i][to_label] = class_predicted

            for i in range(self._n_of_subpaces):
                cfs[i].fit(X[ys[i] != y.dtype.type(-1), idxs[i]],
                           ys[i][ys[i] != y.dtype.type(-1)], **kwards)

        self.h_ = cfs
        self.classes_ = self.h_[0].classes_
        self.columns_ = idxs

        return self


class RelRasco(Rasco):

    def _relevance(X, y):
        pass

    def _tournament(relevance):
        pass


class TriTraining(_BaseCoTraining):
    """TriTraining

    Zhi-Hua Zhou and Ming Li, "Tri-training: exploiting unlabeled data using three classifiers," 
    in <i>IEEE Transactions on Knowledge and Data Engineering</i>, vol. 17, no. 11, pp. 1529-1541, Nov. 2005, 
    doi: 10.1109/TKDE.2005.186.
    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_samples=None, random_state=None):
        self.base_estimator = base_estimator
        self.random_state = check_random_state(random_state)
        self.n_samples = n_samples
        self._n_learner = 3

    def _measure_error(X, y, h1: ClassifierMixin, h2: ClassifierMixin):
        y1 = h1.predict(X)
        y2 = h2.predict(X)
        score1 = accuracy_score(y, y1)
        score2 = accuracy_score(y, y2)

        return 0.5  # TODO Real value

    def _another_hs(hs, index):
        another_hs = []
        for i in range(len(hs)):
            if i != index:
                another_hs.append(hs[i])
        return another_hs

    def _subsample(L, s, random_state=None):
        return resample(*L, replace=False, n_samples=len(L)-s, random_state=random_state)

    def fit(self, X, y, **kwards):
        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        hipotesis = []
        e = [.5]*self._n_learner
        l = [0]*self._n_learner

        updates = [False]*3

        for _ in range(self._n_learner):
            X_sampled, y_sampled = resample(X_label, y_label, replace=True,
                                            n_samples=self.n_samples,
                                            random_state=self.random_state)

            hipotesis.append(
                skclone(self.base_estimator).fit(X_sampled, y_sampled, **kwards)
            )

        something_has_changed = True
        
        while something_has_changed:
            something_has_changed = False
            L = []
            Ly = []
            _e = []

            for i in range(self._n_learner):
                L.append([])
                Ly.append([])
                updates[i] = False
                hj, hk = TriTraining._another_hs(hipotesis, i)
                _e.append(TriTraining._measure_error(X_label, y_label, hj, hk))
                if e[i] > _e[i]:
                    y_p = hj(X_unlabel)
                    validx = yp == hk(X_unlabel)
                    L[i] = X_unlabel[validx]
                    Ly[i] = y_p[validx]
                    
                    if l[i] == 0:
                        l[i] = math.floor(_e[i]/(e[i]-_e[i])+1)
                    if l[i] < len(L[i]):
                        if _e[i]*len(L[i]) < e[i]*l[i]:
                            updates[i] = True
                        elif l[i] > (_e[i]/(e[i]-_e[i])):
                            L[i], Ly[i] = TriTraining._subsample((L[i], Ly[i]), math.ceil(e[i]*l[i]/_e[i]-1), self.random_state)
                            updates[i] = True

            for i in range(self._n_learning):
                if updates[i]:
                    _tempL = np.concatenate(X_label, L[i])
                    _tempY = np.concatenate(y_label, Ly[i])
                    hipotesis[i].fit(_tempL, _tempY, **kwards)
                    e[i] = _e[i]
                    l[i] = len(L[i])
                    something_has_changed = True

        self.h_ = hipotesis
        self.classes_ = self.h_[0].classes_

        return self


class CoTrainingByCommittee(ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, ensemble_estimator=BaggingClassifier(),
                 max_iterations=100, poolsize=100, random_state=None):
        """Create a committee trained by cotraining based on
        the diversity of classifiers.

        Parameters
        ----------
        ensemble_estimator : ClassifierMixin, optional
            ensemble method, works without a ensemble as
            self training with pool, by default BaggingClassifier().
        max_iterations : int, optional
            number of iterations of training, -1 if no max iterations, by default 100
        poolsize : int, optional
            max number of unlabel instances candidates to pseudolabel, by default 100
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        assert isinstance(ensemble_estimator, ClassifierMixin), "This method only support classification"
        self.ensemble_estimator = skclone(ensemble_estimator)
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.random_state = check_random_state(random_state)

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
        prior = self._calculate_prior(y_label)
        permutation = self.random_state.permutation(len(X_unlabel))

        self.ensemble_estimator.fit(X_label, y_label, **kwards)
        self.classes_ = self.ensemble_estimator.classes_

        for _ in range(self.max_iterations):
            if len(permutation) == 0:
                break
            raw_predictions = self.ensemble_estimator.predict_proba(X_unlabel[permutation[0:self.poolsize]])

            predictions = np.max(raw_predictions, axis=1)
            class_predicted = np.argmax(raw_predictions, axis=1)

            to_label = None
            for c in range(len(self.classes_)):
                to_add = np.logical_and(class_predicted == c, predictions >= prior[self.classes_[c]])
                if to_label is not None:
                    to_label = np.logical_or(to_label, to_add)
                else:
                    to_label = to_add

            index = permutation[0:self.poolsize][to_label]
            X_label = np.append(X_label, X_unlabel[index], axis=0)
            pseudoy = np.array(list(map(lambda x: self.classes_[x], class_predicted[to_label])))
            y_label = np.append(y_label, pseudoy, axis=0)
            permutation = permutation[list(map(lambda x: x not in index, permutation))]

            self.ensemble_estimator.fit(X_label, y_label, **kwards)

        return self

    def _calculate_prior(self, y):
        """Calculate the priori probability of each label

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            array of labels

        Returns
        -------
        class_probability: dict
            dictionary with priori probability (value) of each label (key)
        """
        unique, counts = np.unique(y, return_counts=True)
        u_c = dict(zip(unique, counts))
        instances = len(y)
        for u in u_c:
            u_c[u] = float(u_c[u]/instances)
        return u_c

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
        check_is_fitted(self.ensemble_estimator)
        return self.ensemble_estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.

        The predicted class probability depends on the ensemble estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y: ndarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1
            The predicted classes
        """
        check_is_fitted(self.ensemble_estimator)
        return self.ensemble_estimator.predict_proba(X)

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
