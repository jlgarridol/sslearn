import math
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin
from sklearn.base import clone as skclone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state, resample

from ..base import get_dataset
from ..restricted import WhoIsWhoClassifier, combine_predictions
from ..utils import check_n_jobs, safe_division
from ._co import _BaseCoTraining


class TriTraining(_BaseCoTraining):

    def __init__(
        self,
        base_estimator=DecisionTreeClassifier(),
        n_samples=None,
        random_state=None,
        n_jobs=None,
    ):
        """TriTraining
        Zhi-Hua Zhou and Ming Li,
        "Tri-training: exploiting unlabeled data using three classifiers,"
        in <i>IEEE Transactions on Knowledge and Data Engineering</i>,
        vol. 17, no. 11, pp. 1529-1541, Nov. 2005,
        doi: 10.1109/TKDE.2005.186.
        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        n_samples : int, optional
            Number of samples to generate.
            If left to None this is automatically set to the first dimension of the arrays., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = base_estimator
        self.n_samples = n_samples
        self._N_LEARNER = 3
        self._epsilon = sys.float_info.epsilon
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwards):
        """Build a TriTraining classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabeled.
        Returns
        -------
        self: TriTraining
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        self.n_jobs = check_n_jobs(self.n_jobs)

        X_label, y_label, X_unlabel = get_dataset(X, y)

        hypotheses = []
        e_ = [0.5] * self._N_LEARNER
        l_ = [0] * self._N_LEARNER

        # Get a random instance for each class to keep class index
        classes = set(np.unique(y_label))
        instances = list()
        labels = list()
        for x_, y_ in zip(X_label, y_label):
            if y_ in classes:
                classes.remove(y_)
                instances.append(x_)
                labels.append(y_)
            if len(classes) == 0:
                break

        for _ in range(self._N_LEARNER):
            X_sampled, y_sampled = resample(
                X_label,
                y_label,
                replace=True,
                n_samples=self.n_samples,
                random_state=random_state,
            )

            X_sampled = np.concatenate((np.array(instances), X_sampled), axis=0)
            y_sampled = np.concatenate((np.array(labels), y_sampled), axis=0)

            hypotheses.append(
                skclone(self.base_estimator).fit(X_sampled, y_sampled, **kwards)
            )

        something_has_changed = True

        while something_has_changed:
            something_has_changed = False
            L = [[]] * self._N_LEARNER
            Ly = [[]] * self._N_LEARNER
            e = []
            updates = [False] * 3

            for i in range(self._N_LEARNER):
                hj, hk = TriTraining._another_hs(hypotheses, i)
                e.append(
                    self._measure_error(X_label, y_label, hj, hk, self._epsilon)
                )
                if e_[i] <= e[i]:
                    continue
                y_p = hj.predict(X_unlabel)
                validx = y_p == hk.predict(X_unlabel)
                L[i] = X_unlabel[validx]
                Ly[i] = y_p[validx]

                if l_[i] == 0:
                    l_[i] = math.floor(
                        safe_division(e[i], (e_[i] - e[i]), self._epsilon) + 1
                    )
                if l_[i] >= len(L[i]):
                    continue
                if e[i] * len(L[i]) < e_[i] * l_[i]:
                    updates[i] = True
                elif l_[i] > safe_division(e[i], e_[i] - e[i], self._epsilon):
                    L[i], Ly[i] = TriTraining._subsample(
                        (L[i], Ly[i]),
                        math.ceil(
                            safe_division(e_[i] * l_[i], e[i], self._epsilon) - 1
                        ),
                        random_state,
                    )
                    updates[i] = True

            hypotheses = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_estimator)(
                    hypotheses[i], X_label, y_label, L[i], Ly[i], updates[i], **kwards
                )
                for i in range(self._N_LEARNER)
            )

            for i in range(self._N_LEARNER):
                if updates[i]:
                    e_[i] = e[i]
                    l_[i] = len(L[i])
                    something_has_changed = True

        self.h_ = hypotheses
        self.classes_ = self.h_[0].classes_
        self.columns_ = [list(range(X.shape[1]))] * self._N_LEARNER

        return self

    def _fit_estimator(self, hyp, X_label, y_label, L, Ly, update, **kwards):
        if update:
            _tempL = np.concatenate((X_label, L))
            _tempY = np.concatenate((y_label, Ly))

            return hyp.fit(_tempL, _tempY, **kwards)
        return hyp

    @staticmethod
    def _another_hs(hs, index):
        """Get the other hypotheses
        Parameters
        ----------
        hs : list
            hypotheses collection
        index : int
            base hypothesis  index
        Returns
        -------
        list
        """
        another_hs = []
        for i in range(len(hs)):
            if i != index:
                another_hs.append(hs[i])
        return another_hs

    @staticmethod
    def _subsample(L, s, random_state=None):
        """Randomly removes |L| - s number of examples from L
        Parameters
        ----------
        L : tuple of array-like
            Collection pseudo-labeled candidates and its labels
        s : int
            Equation 10 in paper
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        Returns
        -------
        tuple
            Collection of pseudo-labeled selected for enlarged labeled examples.
        """
        to_remove = len(L[0]) - s
        select = len(L[0]) - to_remove

        return resample(*L, replace=False, n_samples=select, random_state=random_state)

    def _measure_error(
        self, X, y, h1: ClassifierMixin, h2: ClassifierMixin, epsilon=sys.float_info.epsilon, **kwards
    ):
        """Calculate the error between two hypotheses
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training labeled input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        h1 : ClassifierMixin
            First hypothesis
        h2 : ClassifierMixin
            Second hypothesis
        epsilon : float
            A small number to avoid division by zero
        Returns
        -------
        float
            Division of the number of labeled examples on which both h1 and h2 make incorrect classification,
            by the number of labeled examples on which the classification made by h1 is the same as that made by h2.
        """
        y1 = h1.predict(X)
        y2 = h2.predict(X)

        error = np.count_nonzero(np.logical_and(y1 == y2, y2 != y))
        coincidence = np.count_nonzero(y1 == y2)
        return safe_division(error, coincidence, epsilon)


class WiWTriTraining(TriTraining):

    def __init__(
        self,
        base_estimator,
        n_samples = 100,
        n_jobs = None,
        method="hungarian", 
        conflict_weighted=True,
        conflict_over="labeled",
        random_state = None,
    ):
        """TriTraining with restriction Who-is-Who.

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        n_samples : int, optional
            Number of samples to generate.
            If left to None this is automatically set to the first dimension of the arrays., by default None
        n_jobs : int, optional
           Number of jobs to run in parallel. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors., by default None
        method : str, optional
            The method to use to assing class, it can be `greedy` to first-look or `hungarian` to use the Hungarian algorithm, by default "hungarian"
        conflict_weighted : bool, default=True
            Whether to weighted the confusion rate by the number of instances with the same group.
        conflict_over : str, optional
            The conflict rate penalizes the "measure error" of basic TriTraining, it can be calculated over differentes subsamples of X, can be:
            * "labeled" over complete L,
            * "labeled_plus" over complete L union L',
            * "unlabeled¨: over complete U,
            * "all": over complete X (LuU) and
            * "none": don't penalize the "meause error", by default "labeled"
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """    
        super().__init__(base_estimator, n_samples, random_state, n_jobs)
        conflict_over_choices = ["labeled", "labeled_plus", "unlabeled", "all", "none"]
        if conflict_over not in conflict_over_choices:
            raise ValueError(
                f"conflict_over must be one of {conflict_over_choices}, got {conflict_over}"
            )        
        self.conflict_over = conflict_over
        self.method=method
        self.conflict_weighted=conflict_weighted

    def fit(self, X, y, instance_group, **kwards):
        """Build a TriTraining classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabeled.
        instance_group : array-like of shape (n_samples)
            The group. Two instances with the same label are not allowed to be in the same group.
        Returns
        -------
        self: TriTraining
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        self.n_jobs = check_n_jobs(self.n_jobs)

        X_label, y_label, X_unlabel = get_dataset(X, y)
        group_label = instance_group[y != y.dtype.type(-1)]
        group_unlabel = instance_group[y == y.dtype.type(-1)]

        hypotheses = []
        e_ = [0.5] * self._N_LEARNER
        l_ = [0] * self._N_LEARNER

        # Get a random instance for each class to keep class index
        classes = set(np.unique(y_label))
        instances = list()
        labels = list()
        groups = list()
        for x_, y_, g_ in zip(X_label, y_label, group_label):
            if y_ in classes:
                classes.remove(y_)
                instances.append(x_)
                labels.append(y_)
                groups.append(g_)
            if len(classes) == 0:
                break

        for _ in range(self._N_LEARNER):
            X_sampled, y_sampled, group_sample = resample(
                X_label,
                y_label,
                group_label,
                replace=False, # It must be False to keep the group restriction
                n_samples=self.n_samples,
                random_state=random_state,
            )

            X_sampled = np.concatenate((np.array(instances), X_sampled), axis=0)
            y_sampled = np.concatenate((np.array(labels), y_sampled), axis=0)
            group_sample = np.concatenate((np.array(groups), group_sample), axis=0)

            hypotheses.append(
                WhoIsWhoClassifier(self.base_estimator, method=self.method, conflict_weighted=self.conflict_weighted).
                        fit(X_sampled, y_sampled, instance_group=group_sample, **kwards)
            )

        something_has_changed = True

        while something_has_changed:
            something_has_changed = False
            L = [[]] * self._N_LEARNER
            Ly = [[]] * self._N_LEARNER
            G = [[]] * self._N_LEARNER
            e = []
            updates = [False] * 3

            for i in range(self._N_LEARNER):
                hj, hk = TriTraining._another_hs(hypotheses, i)
                e.append(
                    self._measure_error(X_label, y_label, hj, hk, self._epsilon, U=X_unlabel, LU=X, GL=group_label, GU=group_unlabel, GLU=instance_group)
                )
                if e_[i] <= e[i]:
                    continue
                y_p = hj.predict(X_unlabel, instance_group=group_unlabel)
                validx = y_p == hk.predict(X_unlabel, instance_group=group_unlabel)
                L[i] = X_unlabel[validx]
                Ly[i] = y_p[validx]
                G[i] = group_unlabel[validx]

                if l_[i] == 0:
                    l_[i] = math.floor(
                        safe_division(e[i], (e_[i] - e[i]), self._epsilon) + 1
                    )
                if l_[i] >= len(L[i]):
                    continue
                if e[i] * len(L[i]) < e_[i] * l_[i]:
                    updates[i] = True
                elif l_[i] > safe_division(e[i], e_[i] - e[i], self._epsilon):
                    L[i], Ly[i], G[i] = TriTraining._subsample(
                        (L[i], Ly[i], G[i]),
                        math.ceil(
                            safe_division(e_[i] * l_[i], e[i], self._epsilon) - 1
                        ),
                        random_state,
                    )
                    updates[i] = True

            hypotheses = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_estimator)(
                    hypotheses[i], X_label, y_label, L[i], Ly[i], updates[i], group_label=group_label, Lg=G[i], **kwards
                )
                for i in range(self._N_LEARNER)
            )

            for i in range(self._N_LEARNER):
                if updates[i]:
                    e_[i] = e[i]
                    l_[i] = len(L[i])
                    something_has_changed = True

        self.h_ = hypotheses
        self.classes_ = self.h_[0].classes_
        self.columns_ = [list(range(X.shape[1]))] * self._N_LEARNER

        return self

    def _measure_error(self, L, y, h1: ClassifierMixin, h2: ClassifierMixin, epsilon=sys.float_info.epsilon, **kwards):
        """Calculate the error between two hypotheses
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training labeled input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        h1 : ClassifierMixin
            First hypothesis
        h2 : ClassifierMixin
            Second hypothesis
        epsilon : float
            A small number to avoid division by zero
        Returns
        -------
        float
            Division of the number of labeled examples on which both h1 and h2 make incorrect classification,
            by the number of labeled examples on which the classification made by h1 is the same as that made by h2.
        """
        me = super()._measure_error(L, y, h1.base_estimator, h2.base_estimator, epsilon)

        LU = kwards.get("LU", None)
        U = kwards.get("U", None)
        GL = kwards.get("GL", None)
        GU = kwards.get("GU", None)
        GLU = kwards.get("GLU", None)
        if self.conflict_over == "labeled":
            conflict = (h1.conflict_rate(L, GL) + h2.conflict_rate(L, GL))
        elif self.conflict_over == "labeled_plus":
            conflict = (h1.conflict_in_train + h2.conflict_in_train)
        elif self.conflict_over == "unlabeled":
            conflict = (h1.conflict_rate(U, GU) + h2.conflict_rate(U, GU))
        elif self.conflict_over == "all":
            conflict = (h1.conflict_rate(LU, GLU) + h2.conflict_rate(LU, GLU))
        else:
            conflict = 0
        return me * (1 + conflict/2)

    def _fit_estimator(self, hyp, X_label, y_label, L, Ly, update, **kwards):
        Lg = kwards.pop("Lg", None)
        group_label = kwards.pop("group_label", None)
        kwards
        if update:
            _tempL = np.concatenate((X_label, L))
            _tempY = np.concatenate((y_label, Ly))
            _tempG = np.concatenate((group_label, Lg))

            return hyp.fit(_tempL, _tempY, instance_group=_tempG, **kwards)
        return hyp
    
    def predict(self, X, instance_group):
        y_probas = self.predict_proba(X)

        y_preds = combine_predictions(y_probas, instance_group, len(self.classes_), self.method)

        return self.classes_.take(y_preds)
