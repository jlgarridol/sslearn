import math
import sys
import warnings as warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import ClassifierMixin
from sklearn.base import clone as skclone
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state, resample
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning

from ..base import get_dataset
from ..restricted import WhoIsWhoClassifier, combine_predictions
from ..utils import check_classifier, check_n_jobs, safe_division
from ._co import BaseCoTraining

import time


class TriTraining(BaseCoTraining):
    """
    **TriTraining. Trio of classifiers with bootstrapping.**

    The main process is:
    1. Generate three classifiers using bootstrapping.
    2. Iterate until convergence:
        1. Calculate the error between two hypotheses.
        2. If the error is less than the previous error, generate a dataset with the instances where both hypotheses agree.
        3. Retrain the classifiers with the new dataset and the original labeled dataset.
    3. Combine the predictions of the three classifiers.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **References**
    ----------
    Zhi-Hua Zhou and Ming Li,<br>
    Tri-training: exploiting unlabeled data using three classifiers,<br>
    in <i>IEEE Transactions on Knowledge and Data Engineering</i>,<br>
    vol. 17, no. 11, pp. 1529-1541, Nov. 2005,<br>
    [10.1109/TKDE.2005.186](https://doi.org/10.1109/TKDE.2005.186)

    """

    def __init__(
        self,
        base_estimator=DecisionTreeClassifier(),
        n_samples=None,
        random_state=None,
        n_jobs=None,
    ):
        """TriTraining. Trio of classifiers with bootstrapping.

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        n_samples : int, optional
            Number of samples to generate.
            If left to None this is automatically set to the first dimension of the arrays., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        n_jobs : int, optional
            The number of jobs to run in parallel for both `fit` and `predict`.
            `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
            `-1` means using all processors., by default None
       
        """
        self._N_LEARNER = 3
        self.base_estimator = check_classifier(base_estimator, collection_size=self._N_LEARNER)
        self.n_samples = n_samples
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
        self : TriTraining
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        self.n_jobs = min(check_n_jobs(self.n_jobs), self._N_LEARNER)

        X_label, y_label, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        hypotheses = []
        e_ = [0.5] * self._N_LEARNER
        l_ = [0] * self._N_LEARNER

        # Get a random instance for each class to keep class index
        self.classes_ = np.unique(y_label)
        classes = set(self.classes_)
        instances = list()
        labels = list()
        iteration = zip(X_label, y_label)
        if is_df:
            iteration = zip(X_label.values, y_label)
        for x_, y_ in iteration:
            if y_ in classes:
                classes.remove(y_)
                instances.append(x_)
                labels.append(y_)
            if len(classes) == 0:
                break

        for i in range(self._N_LEARNER):
            X_sampled, y_sampled = resample(
                X_label,
                y_label,
                replace=True,
                n_samples=self.n_samples,
                random_state=random_state,
            )

            if is_df:
                X_sampled = pd.DataFrame(X_sampled, columns=X_label.columns)
                X_sampled = pd.concat([pd.DataFrame(instances, columns=X_label.columns), X_sampled])
            else:
                X_sampled = np.concatenate((np.array(instances), X_sampled), axis=0)
            y_sampled = np.concatenate((np.array(labels), y_sampled), axis=0)

            hypotheses.append(
                skclone(self.base_estimator if type(self.base_estimator) is not list else self.base_estimator[i]).fit(X_sampled, y_sampled, **kwards)
            )

        something_has_changed = True if X_unlabel.size > 0 else False
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
                    if is_df:
                        L[i] = pd.DataFrame(L[i], columns=X_label.columns)
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
        self.columns_ = [list(range(X.shape[1]))] * self._N_LEARNER

        return self

    def _fit_estimator(self, hyp, X_label, y_label, L, Ly, update, **kwards):
        if update:
            if isinstance(L, pd.DataFrame):
                _tempL = pd.concat([X_label, L])
            else:
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
        classifiers: list
            Collection of other hypotheses
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
        subsamples: tuple
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
        error : float
            Division of the number of labeled examples on which both h1 and h2 make incorrect classification,
            by the number of labeled examples on which the classification made by h1 is the same as that made by h2.
        """
        y1 = h1.predict(X)
        y2 = h2.predict(X)

        error = np.count_nonzero(np.logical_and(y1 == y2, y2 != y))
        coincidence = np.count_nonzero(y1 == y2)
        return safe_division(error, coincidence, epsilon)


class WiWTriTraining(TriTraining):
    """
    **Who-Is-Who TriTraining.**
     
    Trio of classifiers with bootstrapping and restricted set classification.
    Is the same as TriTraining but with the restricted set classification.
    Maninly, the conflict rate penalizes the ***measure error*** of basic TriTraining, it can be calculated over differentes subsamples of X, can be:
    * `labeled` over complete L,
    * `labeled_plus` over complete L union L',
    * `unlabeled`: over complete U,
    * `all`: over complete X (LuU) and
    * `none`: don't penalize the ***meause error***, only use the restrictions for avoid share classes in the same group. 
    
    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances. Receives the instance group, an array-like of shape (n_samples) with the group of each instance. Two instances with the same label are not allowed to be in the same group.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **References**
    ----------
    Ludmila I. Kuncheva, Juan J. Rodríguez, Aaron S. Jackson, (2016)<br>
    Restricted set classification: Who is there?<br>
    <i>Pattern Recognition</i>, 63, 158-170, <br>
    [10.1016/j.patcog.2016.08.028](https://doi.org/10.1016/j.patcog.2016.08.028)
    """

    def __init__(
        self,
        base_estimator,
        n_samples=100,
        n_jobs=None,
        method="hungarian",
        conflict_weighted=True,
        conflict_over="labeled",
        random_state=None,
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
        self.method = method
        self.conflict_weighted = conflict_weighted

    def fit(self, X, y, instance_group=None, **kwards):
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
        self : TriTraining
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        self.n_jobs = check_n_jobs(self.n_jobs)

        if instance_group is None:
            warn(
                "Instance group is not provided. Each instance will belong their own group. Consider using `TriTraining`."
            )
            instance_group = np.arange(len(y))

        X_label, y_label, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        group_label = instance_group[y != y.dtype.type(-1)]
        group_unlabel = instance_group[y == y.dtype.type(-1)]

        hypotheses = []
        e_ = [0.5] * self._N_LEARNER
        l_ = [0] * self._N_LEARNER

        # Get a random instance for each class to keep class index
        self.classes_ = np.unique(y_label)
        classes = set(self.classes_)
        instances = list()
        labels = list()
        groups = list()
        iteration = zip(X_label, y_label, group_label)
        if is_df:
            iteration = zip(X_label.values, y_label, group_label)
        for x_, y_, g_ in iteration:
            if y_ in classes:
                classes.remove(y_)
                instances.append(x_)
                labels.append(y_)
                groups.append(g_)
            if len(classes) == 0:
                break

        for i in range(self._N_LEARNER):
            X_sampled, y_sampled, group_sample = resample(
                X_label,
                y_label,
                group_label,
                replace=False,  # It must be False to keep the group restriction
                n_samples=self.n_samples,
                random_state=random_state,
            )

            if is_df:
                X_sampled = pd.DataFrame(X_sampled, columns=X_label.columns)
                X_sampled = pd.concat([pd.DataFrame(instances, columns=X_label.columns), X_sampled])
            else:
                X_sampled = np.concatenate((np.array(instances), X_sampled), axis=0)
            y_sampled = np.concatenate((np.array(labels), y_sampled), axis=0)
            group_sample = np.concatenate((np.array(groups), group_sample), axis=0)

            hypotheses.append(
                WhoIsWhoClassifier(self.base_estimator if not isinstance(self.base_estimator, list) else self.base_estimator[i], method=self.method, conflict_weighted=self.conflict_weighted).
                fit(X_sampled, y_sampled, instance_group=group_sample, **kwards)
            )

        something_has_changed = True if X_unlabel.shape[0] > 0 else False

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
                    if is_df:
                        L[i] = pd.DataFrame(L[i], columns=X_label.columns)

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
        error: float
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
        return me * (1 + conflict / 2)

    def _fit_estimator(self, hyp, X_label, y_label, L, Ly, update, **kwards):
        Lg = kwards.pop("Lg", None)
        group_label = kwards.pop("group_label", None)
        kwards
        if update:
            if isinstance(X_label, pd.DataFrame):
                _tempL = pd.concat([X_label, L])
            else:
                _tempL = np.concatenate((X_label, L))
            _tempY = np.concatenate((y_label, Ly))
            _tempG = np.concatenate((group_label, Lg))

            return hyp.fit(_tempL, _tempY, instance_group=_tempG, **kwards)
        return hyp

    def predict(self, X, instance_group):
        y_probas = self.predict_proba(X)

        y_preds = combine_predictions(y_probas, instance_group, len(self.classes_), self.method)

        return self.classes_.take(y_preds)


class DeTriTraining(TriTraining):
    """
    **TriTraining with Data Editing.**

    It is a variation of the TriTraining, the main difference is that the instances are depurated in each iteration.
    It means that the instances with their neighbors that have the same class are kept, the rest are removed.
    At the end of the iterations, the instances are clustered and the class is assigned to the cluster centroid.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **References**
    ----------
    Deng C., Guo M.Z. (2006)<br>
    Tri-training and Data Editing Based Semi-supervised Clustering Algorithm, <br>
    in <i>Gelbukh A., Reyes-Garcia C.A. (eds) MICAI 2006: Advances in Artificial Intelligence. MICAI 2006.</i><br>
    Lecture Notes in Computer Science, vol 4293. Springer, Berlin, Heidelberg.<br>
    [10.1007/11925231_61](https://doi.org/10.1007/11925231_61)
    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), k_neighbors=3,
                 n_samples=None, mode="seeded", max_iterations=100, n_jobs=None, random_state=None):
        """
        DeTriTraining - TriTraining with Depurated and Clustering.
        Avoid the noise generated by the TriTraining algorithm by depurating the enlarged dataset and clustering the instances.        

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        n_samples : int, optional
            Number of samples to generate. 
            If left to None this is automatically set to the first dimension of the arrays., by default None
        k_neighbors : int, optional
            Number of neighbors for depurate classification. 
            If at least k_neighbors/2+1 have a class other than the one predicted, the class is ignored., by default 3
        mode : string, optional
            How to calculate the cluster each instance belongs to.
            If `seeded` each instance belong to nearest cluster.
            If `constrained` each instance belong to nearest cluster unless the instance is in to enlarged dataset, 
            then the instance belongs to the cluster of its class., by default `seeded`
        max_iterations : int, optional
            Maximum number of iterations, by default 100
        n_jobs : int, optional
            The number of parallel jobs to run for neighbors search. 
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. 
            Doesn't affect fit method., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        super().__init__(base_estimator, n_samples, random_state)
        self.k_neighbors = k_neighbors
        self.mode = mode
        self.max_iterations = max_iterations
        self.n_jobs = n_jobs
        if mode != "seeded" and mode != "constrained":
            raise AttributeError("`mode` must be \"seeded\" or \"constrained\".")

    def _depure(self, S):
        """Depure the S dataset

        Parameters
        ----------
        S : tuple (X, y)
            Enlarged dataset

        Returns
        -------
        tuple : (X, y)
            Enlarged dataset with instances where at least k_neighbors/2+1 have the same class.
        """
        init = time.time()
        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors, n_jobs=self.n_jobs)
        valid = knn.fit(*S).predict(S[0]) == S[1]
        print(f"Depure time: {time.time() - init}")
        return S[0][valid], S[1][valid]

    def _clustering(self, S, X):
        """Clustering phase of the fitting

        Parameters
        ----------
        S : tuple (X, y)
            Enlarged dataset
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Complete dataset, only features

        Returns
        -------
        y: array-like of shape (n_samples,)
            class predicted for each instance
        """
        centroids = dict()
        clusters = set(S[1])

        # uses as numpy
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(S[0], pd.DataFrame):
            S = (S[0].to_numpy(), S[1])

        for k in clusters:
            centroids[k] = np.mean(S[0][S[1] == k], axis=0)

        def seeded(X):
            # For each instance, calculate the distance to each centroid
            distances = np.linalg.norm(X[:, None, :] - np.array(list(centroids.values())), axis=2)
            # Get the index of the nearest centroid
            return np.argmin(distances, axis=1)

        def constrained(X):
            # Calculate the distances to centroids using broadcasting
            distances = np.linalg.norm(X[:, None, :] - np.array(list(centroids.values())), axis=2)
            # Get the index of the nearest centroid
            nearest = np.argmin(distances, axis=1)
            # Create a mask to find instances in X that belong to S[0]
            mask = (S[0] == X[:, None])
            # Find the row and column indices where all elements are True
            i, j = np.where(mask.all(axis=2))
            # Initialize cluster with -1
            cluster = np.full(X.shape[0], -1, dtype=int)
            # Update cluster for the instances found in S[0]
            cluster[i] = S[1][j]
            # Update cluster for instances not found in S[0]
            cluster[cluster == -1] = nearest[cluster == -1]

            return cluster

        if self.mode == "seeded":
            op = seeded
        elif self.mode == "constrained":
            op = constrained

        changes = True
        iterations = 0
        while changes and iterations < self.max_iterations:
            changes = False
            iterations += 1
            # Need to vectorize
            new_clusters = op(X)
            new_centroids = dict()
            for k in clusters:
                if np.any(new_clusters == k):
                    new_centroids[k] = np.mean(X[new_clusters == k], axis=0)
                    if not np.array_equal(new_centroids[k], centroids[k]):
                        changes = True
            centroids = new_centroids

        return new_clusters

    def fit(self, X, y, **kwards):
        """Build a DeTriTraining classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabel.

        Returns
        -------
        self: DeTriTraining
            Fitted estimator.
        """
        X_label, y_label, X_unlabel = get_dataset(X, y)

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_label)
        y_label = self.label_encoder_.transform(y_label)

        is_df = isinstance(X_label, pd.DataFrame)

        self.classes_ = np.unique(y_label)

        classes = set(self.classes_)
        instances = list()
        labels = list()
        iteration = zip(X_label, y_label)
        if is_df:
            iteration = zip(X_label.values, y_label)
        for x_, y_ in iteration:
            if y_ in classes:
                classes.remove(y_)
                instances.append(x_)
                labels.append(y_)
            if len(classes) == 0:
                break

        S_ = []
        hypothesis = []
        for i in range(self._N_LEARNER):
            X_sampled, y_sampled = \
                resample(X_label, y_label, replace=True,
                         n_samples=self.n_samples,
                         random_state=self.random_state)
            if is_df:
                X_sampled = pd.DataFrame(X_sampled, columns=X_label.columns)
            hypothesis.append(
                skclone(self.base_estimator if type(self.base_estimator) is not list else self.base_estimator[i]).fit(
                    X_sampled, y_sampled, **kwards)
            )

            # Keep class order
            if not is_df:
                X_sampled = np.concatenate((np.array(instances), X_sampled), axis=0)
            else:
                X_sampled = pd.concat([pd.DataFrame(instances, columns=X_label.columns), X_sampled], axis=0)

            y_sampled = np.concatenate((np.array(labels), y_sampled), axis=0)

            S_.append((X_sampled, y_sampled))

        changes = True
        last_addition = [0] * self._N_LEARNER
        it = 0 if X_unlabel.shape[0] > 0 else self.max_iterations
        while it < self.max_iterations:
            it += 1
            changes = False

            # Enlarged
            L = [[]] * self._N_LEARNER

            for i in range(self._N_LEARNER):
                hj, hk = TriTraining._another_hs(hypothesis, i)
                y_p = hj.predict(X_unlabel)
                validx = y_p == hk.predict(X_unlabel)
                L[i] = (X_unlabel[validx] if not is_df else X_unlabel.iloc[validx, :], y_p[validx])

            for i, _ in enumerate(L):

                if len(L[i][0]) > 0:
                    S_[i] = np.concatenate((X_label, L[i][0])) if not is_df else pd.concat([X_label, L[i][0]]), np.concatenate((y_label, L[i][1]))
                    S_[i] = self._depure(S_[i])

            for i in range(self._N_LEARNER):
                if len(S_[i][0]) > len(X_label):
                    last_addition[i] = len(S_[i][0])
                    changes = True
                    hypothesis[i].fit(*S_[i], **kwards)

            if not changes:
                break
        else:
            warn.warn("Maximum number of iterations reached before convergence. Consider increasing max_iter to improve the fit.", ConvergenceWarning)

        S = np.concatenate([x[0] for x in S_]) if not is_df else pd.concat([x[0] for x in S_]), np.concatenate([x[1] for x in S_])
        S_0, index_ = np.unique(S[0], axis=0, return_index=True)
        S_1 = S[1][index_]
        S = S_0, S_1
        S = self._depure(S)  # Change, is S - L (only new)

        new_y = self._clustering(S, X)

        self.h_ = [skclone(self.base_estimator if type(self.base_estimator) is not list else self.base_estimator[i]).fit(X, new_y, **kwards) for i in range(self._N_LEARNER)]
        self.columns_ = [list(range(X.shape[1]))]

        return self
