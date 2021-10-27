from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.base import clone as skclone
from ._co import TriTraining
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Done and tested
class DeTriTraining(TriTraining):

    def __init__(self, base_estimator=DecisionTreeClassifier(), k_neighbors=3,
                 n_samples=None, mode="seeded", n_jobs=None, random_state=None):
        """DeTriTraining

        Deng C., Guo M.Z. (2006)
        Tri-training and Data Editing Based Semi-supervised Clustering Algorithm. 
        In: Gelbukh A., Reyes-Garcia C.A. (eds) MICAI 2006: Advances in Artificial Intelligence. MICAI 2006. 
        Lecture Notes in Computer Science, vol 4293.
        Springer, Berlin, Heidelberg.
        https://doi.org/10.1007/11925231_61

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
        n_jobs : int, optional
            The number of parallel jobs to run for neighbors search. 
            None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. 
            Doesn’t affect fit method., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        super().__init__(base_estimator, n_samples, random_state)
        self.k_neighbors = k_neighbors
        self.mode = mode
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
        tuple (X, y)
            Enlarged dataset with instances where at least k_neighbors/2+1 have the same class.
        """
        # k_neighbors +1 to ignore the own instance.
        knn = KNeighborsClassifier(n_neighbors=self.k_neighbors + 1, n_jobs=self.n_jobs)
        valid = knn.fit(*S).predict(S[0]) == S[1]
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
        array-like of shape (n_samples,)
            class predicted for each instance
        """        
        centroids = dict()
        clusters = set(S[1])
        for k in clusters:
            centroids[k] = np.mean(S[0][S[1]==k], axis=0)

        def seeded(x):
            min_ = np.inf
            k_min = None
            for k in centroids:
                candidate = np.linalg.norm(x - centroids[k])
                if candidate < min_ or k_min is None:
                    min_ = candidate
                    k_min = k
            return k_min

        def constrained(x):
            candidate = S[1][(S[0] == x).sum(axis=1) == X.shape[1]]
            if len(candidate) == 0:
                return seeded(x)
            else:
                return candidate[0]

        if self.mode == "seeded":
            op = seeded
        elif self.mode == "constrained":
            op = constrained

        changes = True
        while changes:
            changes = False
            
            new_clusters = np.apply_along_axis(op, 1, X)
            new_centroids = dict()
            for k in clusters:
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
        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        S_ = []
        hypothesis = []
        for i in range(self._N_LEARNER):
            X_sampled, y_sampled = \
                resample(X_label, y_label, replace=True,
                         n_samples=self.n_samples,
                         random_state=self.random_state)
            hypothesis.append(
                skclone(self.base_estimator).fit(
                    X_sampled, y_sampled, **kwards)
            )
            S_.append((X_sampled, y_sampled))

        changes = True
        while changes:
            changes = False

            # Enlarged
            L = [[]] * self._N_LEARNER

            for i in range(self._N_LEARNER):
                hj, hk = TriTraining._another_hs(hypothesis, i)
                y_p = hj.predict(X_unlabel)
                validx = y_p == hk.predict(X_unlabel)
                L[i] = (X_unlabel[validx], y_p[validx])

            for i, _ in enumerate(L):
                if len(L[i][0]) > 0:
                    S_[i] = np.concatenate((X_label, L[i][0])), np.concatenate((y_label, L[i][1]))
                    S_[i] = self._depure(S_[i])

            for i in range(self._N_LEARNER):
                if len(S_[i][0]) > len(X_label):
                    changes = True
                    hypothesis[i].fit(*S_[i], **kwards)

        S = np.concatenate([x[0] for x in S_]), np.concatenate([x[1] for x in S_])
        S_0, index_ = np.unique(S[0], axis=0, return_index=True)
        S_1 = S[1][index_]
        S = S_0, S_1
        S = self._depure(S)  # Change, is S - L (only new)

        new_y = self._clustering(S, X)

        self.h_ = [skclone(self.base_estimator).fit(X, new_y, **kwards)]
        self.classes_ = self.h_[0].classes_
        self.columns_ = [list(range(X.shape[1]))]

        return self

