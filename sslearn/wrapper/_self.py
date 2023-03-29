import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone as skclone
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.utils import check_random_state, resample

from sslearn.utils import calculate_prior_probability, check_classifier

from ..base import get_dataset


class SelfTraining(SelfTrainingClassifier):

    """Self-training. Adaptation of SelfTrainingClassifier from sklearn with data loader compatible.

    This class allows a given supervised classifier to function as a
    semi-supervised classifier, allowing it to learn from unlabeled data. It
    does this by iteratively predicting pseudo-labels for the unlabeled data
    and adding them to the training set.

    The classifier will continue iterating until either max_iter is reached, or
    no pseudo-labels were added to the training set in the previous iteration.

    Read more in the :ref:`User Guide <self_training>`.

    Parameters
    ----------
    base_estimator : estimator object
        An estimator object implementing ``fit`` and ``predict_proba``.
        Invoking the ``fit`` method will fit a clone of the passed estimator,
        which will be stored in the ``base_estimator_`` attribute.

    threshold : float, default=0.75
        The decision threshold for use with `criterion='threshold'`.
        Should be in [0, 1). When using the 'threshold' criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    criterion : {'threshold', 'k_best'}, default='threshold'
        The selection criterion used to select which labels to add to the
        training set. If 'threshold', pseudo-labels with prediction
        probabilities above `threshold` are added to the dataset. If 'k_best',
        the `k_best` pseudo-labels with highest prediction probabilities are
        added to the dataset. When using the 'threshold' criterion, a
        :ref:`well calibrated classifier <calibration>` should be used.

    k_best : int, default=10
        The amount of samples to add in each iteration. Only used when
        `criterion` is k_best'.

    max_iter : int or None, default=10
        Maximum number of iterations allowed. Should be greater than or equal
        to 0. If it is ``None``, the classifier will continue to predict labels
        until no new pseudo-labels are added, or all unlabeled samples have
        been labeled.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    base_estimator_ : estimator object
        The fitted estimator.

    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output. (Taken from the trained
        ``base_estimator_``).

    transduction_ : ndarray of shape (n_samples,)
        The labels used for the final fit of the classifier, including
        pseudo-labels added during fit.

    labeled_iter_ : ndarray of shape (n_samples,)
        The iteration in which each sample was labeled. When a sample has
        iteration 0, the sample was already labeled in the original dataset.
        When a sample has iteration -1, the sample was not labeled in any
        iteration.

    n_iter_ : int
        The number of rounds of self-training, that is the number of times the
        base estimator is fitted on relabeled variants of the training set.

    termination_condition_ : {'max_iter', 'no_change', 'all_labeled'}
        The reason that fitting was stopped.

        - 'max_iter': `n_iter_` reached `max_iter`.
        - 'no_change': no new labels were predicted.
        - 'all_labeled': all unlabeled samples were labeled before `max_iter`
          was reached.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import datasets
    >>> from sklearn.semi_supervised import SelfTrainingClassifier
    >>> from sklearn.svm import SVC
    >>> rng = np.random.RandomState(42)
    >>> iris = datasets.load_iris()
    >>> random_unlabeled_points = rng.rand(iris.target.shape[0]) < 0.3
    >>> iris.target[random_unlabeled_points] = -1
    >>> svc = SVC(probability=True, gamma="auto")
    >>> self_training_model = SelfTrainingClassifier(svc)
    >>> self_training_model.fit(iris.data, iris.target)
    SelfTrainingClassifier(...)

    References
    ----------
    David Yarowsky. 1995. Unsupervised word sense disambiguation rivaling
    supervised methods. In Proceedings of the 33rd annual meeting on
    Association for Computational Linguistics (ACL '95). Association for
    Computational Linguistics, Stroudsburg, PA, USA, 189-196. DOI:
    https://doi.org/10.3115/981658.981684
    """
    _estimator_type = "classifier"

    def __init__(self,
                 base_estimator,
                 threshold=0.75,
                 criterion='threshold',
                 k_best=10,
                 max_iter=10,
                 verbose=False):
        super().__init__(base_estimator, threshold, criterion, k_best, max_iter, verbose)

    def fit(self, X, y):
        """
        Fits this ``SelfTrainingClassifier`` to a dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        y : {array-like, sparse matrix} of shape (n_samples,)
            Array representing the labels. Unlabeled samples should have the
            label -1.

        Returns
        -------
        self : SelfTrainingClassifier
            Returns an instance of self.
        """
        y_adapted = y.copy()
        if y_adapted.dtype.type is str or y_adapted.dtype.type is np.str_:
            y_adapted = y_adapted.astype(object)
            y_adapted[y_adapted == '-1'] = -1
        return super().fit(X, y_adapted)


class Setred(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        base_estimator=KNeighborsClassifier(n_neighbors=3),
        max_iterations=40,
        distance="euclidean",
        poolsize=0.25,
        rejection_threshold=0.05,
        graph_neighbors=1,
        random_state=None,
        n_jobs=None,
    ):
        """
        Li, Ming, and Zhi-Hua Zhou. "SETRED: Self-training with editing."
        Pacific-Asia Conference on Knowledge Discovery and Data Mining.
        Springer, Berlin, Heidelberg, 2005. doi: 10.1007/11430919_71.

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba,, by default DecisionTreeClassifier(), by default KNeighborsClassifier(n_neighbors=3)
        max_iterations : int, optional
            Maximum number of iterations allowed. Should be greater than or equal to 0., by default 40
        distance : str, optional
            The distance metric to use for the graph.
            The default metric is euclidean, and with p=2 is equivalent to the standard Euclidean metric.
            For a list of available metrics, see the documentation of DistanceMetric and the metrics listed in sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
            Note that the “cosine” metric uses cosine_distances., by default "euclidean"
        poolsize : float, optional
            Max number of unlabel instances candidates to pseudolabel, by default 0.25
        rejection_threshold : float, optional
            significance level, by default 0.1
        graph_neighbors : int, optional
            Number of neighbors for each sample., by default 1
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        n_jobs : int, optional
            The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors, by default None
        """
        self.base_estimator = check_classifier(base_estimator, can_be_list=False)
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.distance = distance
        self.rejection_threshold = rejection_threshold
        self.graph_neighbors = graph_neighbors
        self.random_state = random_state
        self.n_jobs = n_jobs

    def __create_neighborhood(self, X):
        # kneighbors_graph(X, 1, metric=self.distance, n_jobs=self.n_jobs).toarray()
        return kneighbors_graph(
            X, self.graph_neighbors, metric=self.distance, n_jobs=self.n_jobs, mode="distance"
        ).toarray()

    def fit(self, X, y, **kwars):
        """Build a Setred classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabeled.

        Returns
        -------
        self: Setred
            Fitted estimator.
        """        
        random_state = check_random_state(self.random_state)

        X_label, y_label, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        self.classes_ = np.unique(y_label)

        each_iteration_candidates = X_label.shape[0]

        pool = int(len(X_unlabel) * self.poolsize)
        self._base_estimator = skclone(self.base_estimator)

        self._base_estimator.fit(X_label, y_label, **kwars)

        y_probabilities = calculate_prior_probability(
            y_label
        )  # Should probabilities change every iteration or may it keep with the first L?

        sort_idx = np.argsort(list(y_probabilities.keys()))

        if X_unlabel.shape[0] == 0:
            return self

        for _ in range(self.max_iterations):
            U_ = resample(
                X_unlabel, replace=False, n_samples=pool, random_state=random_state
            )

            if is_df:
                U_ = pd.DataFrame(U_, columns=X_label.columns)

            raw_predictions = self._base_estimator.predict_proba(U_)
            predictions = np.max(raw_predictions, axis=1)
            class_predicted = np.argmax(raw_predictions, axis=1)
            # Unless a better understanding is given, only the size of L will be used as maximal size of the candidate set.
            indexes = predictions.argsort()[-each_iteration_candidates:]

            if is_df:
                L_ = U_.iloc[indexes]
            else:
                L_ = U_[indexes]
            y_ = np.array(
                list(
                    map(
                        lambda x: self._base_estimator.classes_[x],
                        class_predicted[indexes],
                    )
                )
            )

            if is_df:
                pre_L = pd.concat([X_label, L_])
            else:
                pre_L = np.concatenate((X_label, L_), axis=0)

            weights = self.__create_neighborhood(pre_L)
            #  Keep only weights for L_
            weights = weights[-L_.shape[0]:, :]

            idx = np.searchsorted(np.array(list(y_probabilities.keys())), y_, sorter=sort_idx)
            p_wrong = 1 - np.asarray(np.array(list(y_probabilities.values())))[sort_idx][idx]
            #  Must weights be the inverse of distance?
            weights = np.divide(1, weights, out=np.zeros_like(weights), where=weights != 0)

            weights_sum = weights.sum(axis=1)
            weights_square_sum = (weights ** 2).sum(axis=1)

            iid_random = random_state.binomial(
                1, np.repeat(p_wrong, weights.shape[1]).reshape(weights.shape)
            )
            ji = (iid_random * weights).sum(axis=1)

            mu_h0 = p_wrong * weights_sum
            sigma_h0 = np.sqrt((1 - p_wrong) * p_wrong * weights_square_sum)
            
            z_score = np.divide((ji - mu_h0), sigma_h0, out=np.zeros_like(sigma_h0), where=sigma_h0 != 0)
            # z_score = (ji - mu_h0) / sigma_h0
            
            oi = norm.sf(abs(z_score), mu_h0, sigma_h0)
            to_add = (oi < self.rejection_threshold) & (z_score < mu_h0)

            if is_df:
                L_filtered = L_.iloc[to_add, :]
            else:
                L_filtered = L_[to_add, :]
            y_filtered = y_[to_add]
            
            if is_df:
                X_label = pd.concat([X_label, L_filtered])
            else:
                X_label = np.concatenate((X_label, L_filtered), axis=0)
            y_label = np.concatenate((y_label, y_filtered), axis=0)

            #  Remove the instances from the unlabeled set.
            to_delete = indexes[to_add]
            if is_df:
                X_unlabel = X_unlabel.drop(index=X_unlabel.index[to_delete])
            else:
                X_unlabel = np.delete(X_unlabel, to_delete, axis=0)

        return self

    def predict(self, X, **kwards):
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
        return self._base_estimator.predict(X, **kwards)

    def predict_proba(self, X, **kwards):
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
        return self._base_estimator.predict_proba(X, **kwards)
