import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import (check_X_y,
                           check_random_state, resample)
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone as skclone
from sklearn.utils.validation import check_is_fitted
from sktime.utils.validation import check_n_jobs
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder


class Rotation(TransformerMixin, BaseEstimator):

    def __init__(self, min_group=3, max_group=3, remove_proportion=.50, pca=PCA(), normalize=True, random_state=None):
        """Rotation Transformer.
        Join of subspaces of the dataset transfomed with PCA keeping the subspace dimensionality.
        The transformation is made with a subsamples of the subspace. The subsamples and the subspaces are made randomly.

        Implemention inspired on `Rotation Forest` by MatthewMiddlehurst from SkTime Contrib
        (https://raw.githubusercontent.com/alan-turing-institute/sktime/9a43c79752c1db176adc2e4e6060a33ec9a08308/sktime/contrib/vector_classifiers/_rotation_forest.py)
        License BSD-3 Clause

        Parameters
        ----------
        group_size : int, optional
            Number of the features for each subspace.
            If the number of features mod group size are not zero,
            then the last subspace are created with
            features used previously selected randomly., by default 3
        group_weight : float, optional
            Proportion of instances to be removed., by default .50
        pca : PCA, optional
            PCA configuration, the n_components will be overwritten., by default PCA()
        normalize : boolean, optional
            Normalize data before fit and transform, the n_components will be overwritten., by default True
        random_state : None, int or RandomState, optional
            Random state for create subspaces and subsamples., by default None

        References
        ----------
        .. [1] Rodriguez, Juan José, Ludmila I. Kuncheva, and Carlos J. Alonso. "Rotation
        forest: A new classifier ensemble method." IEEE transactions on pattern analysis
        and machine intelligence 28.10 (2006).
        """
        self.min_group = min_group
        self.max_group = max_group
        self.random_state = random_state
        self.remove_proportion = remove_proportion

        pca.n_components = None
        self.pca = pca
        self.normalize = normalize

    def fit(self, X, y=None):
        """Create a rotation.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Training data, where n_samples is the number of samples and n_features
                is the number of features.
            y (None): If not None it used for keep class proportion.
            generate_X (True): Create a fit dataset based on Rotation Forest
        """

        self.n_instances, self.n_atts = X.shape

        self._random_state = check_random_state(self.random_state)

        if y is not None:
            self.classes_ = np.unique(y)
            self.n_classes = self.classes_.shape[0]
            X_cls_split = [X[np.where(y == i)] for i in self.classes_]

        if self.normalize:
            self._calc_normalization(X, self._random_state)
            X = self._normalize(X)

        self.groups_ = self._generate_groups(self._random_state)
        self.pcas_ = []

        for group in self.groups_:

            if y is not None:
                classes = self._random_state.choice(
                    range(self.n_classes),
                    size=self._random_state.randint(1, self.n_classes + 1),
                    replace=False,
                )

                # randomly add the classes with the randomly selected attributes.
                X_t = np.zeros((0, len(group)))
                for cls_idx in classes:
                    c = X_cls_split[cls_idx]
                    X_t = np.concatenate((X_t, c[:, group]), axis=0)
                    # random_state.random_sample((10, X_t.shape[1]))

                original_X = X_t

                samples = self._random_state.choice(
                    X_t.shape[0],
                    int(X_t.shape[0] * (1.0 - self.remove_proportion)),
                    replace=False
                )

                X_t = X_t[samples]
                # X_t = resample(X_t,
                #                replace=False,
                #                n_samples=int(
                #                    (1.0 - self.remove_proportion) * X_t.shape[0]),
                #                random_state=self._random_state.randint(100)
                #                )

            else:
                original_X = X[:, group]
                X_t = resample(X[:, group], replace=False,
                               n_samples=int(
                                   (1.0 - self.remove_proportion) * self.n_instances),
                               random_state=self._random_state.randint(100))

            if X_t.shape[0] < 2:
                X_t = original_X

            while True:
                try:
                    pca = skclone(self.pca).fit(X_t)
                    break
                except Exception:
                    X_t = np.concatenate(
                        (X_t, self._random_state.random_sample((10, X_t.shape[1]))), axis=0
                    )
                    # X_t = np.concatenate(
                    #     (X_t, resample(X_t, replace=False, n_samples=10, random_state=self._random_state.randint(100))), axis=0
                    # )

            self.pcas_.append(pca)

        return self

    def transform(self, X, **kwargs):
        """Apply rotation to X.

        X rotated in each subspace and then the rotated subspaces are joined to create the global rotation of X.

        Args:
            X (array-like, shape (n_samples, n_features)): New data, where n_samples is the number of samples and n_features is the number of features.

        Returns:
            array-like, shape (n_samples, n_components): Transformed values.
        """
        if "pcas_" not in dir(self):
            raise NotFittedError("Fit before transform.")

        if self.normalize:
            X = self._normalize(X)

        return np.concatenate(
            [self.pcas_[i].transform(X[:, group])
                for i, group in enumerate(self.groups_)], axis=1
        )

    def _normalize(self, X):
        return (X - self._med) / (self._std + self._noise)

    def _calc_normalization(self, X, random):
        self._std = np.std(X, axis=0)
        self._med = np.mean(X, axis=0)
        self._noise = [random.uniform(-0.000005, 0.000005) for p in range(0, X.shape[1])]

    def _generate_groups(self, rng):
        permutation = rng.permutation((np.arange(0, self.n_atts)))

        # select the size of each group.
        group_size_count = np.zeros(self.max_group - self.min_group + 1)
        n_attributes = 0
        n_groups = 0
        while n_attributes < self.n_atts:
            n = rng.randint(group_size_count.shape[0])
            group_size_count[n] += 1
            n_attributes += self.min_group + n
            n_groups += 1

        groups = []
        current_attribute = 0
        current_size = 0
        for i in range(0, n_groups):
            while group_size_count[current_size] == 0:
                current_size += 1
            group_size_count[current_size] -= 1

            n = self.min_group + current_size
            groups.append(np.zeros(n, dtype=np.int))
            for k in range(0, n):
                if current_attribute < permutation.shape[0]:
                    groups[i][k] = permutation[current_attribute]
                else:
                    groups[i][k] = permutation[rng.randint(
                        permutation.shape[0])]
                current_attribute += 1

        return groups


class RotatedTree(ClassifierMixin, BaseEstimator):

    def __init__(self, base_estimator=DecisionTreeClassifier(criterion="entropy"), rotation=Rotation()):
        """Create a rotation and train a decision tree classifier.

        Args:
            base_estimator (object, optional): The base estimator to fit on rotation of the dataset. If None, then the base estimator is a decision tree. Defaults to DecisionTreeClassifier().
            rotation (Rotation, optional): The configured rotation transform. Defaults to Rotation().
        """
        self.base_estimator = skclone(base_estimator)
        self.rotation = skclone(rotation)

    def fit(self, X, y):
        """Fit the Rotated Tree model.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like, shape (n_samples,)): The target values.
        """
        X_transformed = self.rotation.fit(X).transform(X, normalize=False)
        self.base_estimator.fit(X_transformed, y)
        self.classes_ = self.base_estimator.classes_

        return self

    def predict(self, X):
        """Predict class for X.

        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            ndarray of shape (n_samples,): The predicted classes.
        """
        X = self.rotation.transform(X)
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.


        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            array-like, shape (n_samples, n_features): The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        """
        X = self.rotation.transform(X)
        return self.base_estimator.predict_proba(X)


class RotationForestClassifier(ClassifierMixin, BaseEnsemble):

    def __init__(self, base_estimator=DecisionTreeClassifier(criterion="entropy"), n_estimators=10, min_group_size=3, max_group_size=3, group_weight=0.5, transformer=PCA(), n_jobs=None, random_state=None):

        self.random_state = random_state
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size
        self.base_estimator = base_estimator
        self.group_weight = group_weight
        self.transformer = skclone(transformer)
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

        self.__check()

    def __check(self):
        if self.min_group_size > self.max_group_size:
            raise AttributeError("Minimum group size ({}) cannot be higher than maximum group size ({})".format(
                self.min_group_size, self.max_group_size))
        if self.group_weight > 1 or self.group_weight < 0:
            raise AttributeError("The remove percentage ({}) cannot be lower than 0 or higher than 100".format(
                self.group_weight))
        if not isinstance(self.transformer, TransformerMixin):
            raise AttributeError("The declared transformer ({}) is not a TransformerMixin".format(
                type(self.transformer)))
        self.n_jobs = check_n_jobs(self.n_jobs)

    def fit(self, X, y):
        """Fit the RotationForest model.

        Args:
            X (array-like, shape (n_samples, n_features)): Training data, where n_samples is the number of samples and n_features is the number of features.
            y (array-like, shape (n_samples,)): The target values.
        """
        random_state = check_random_state(self.random_state)
        X, y = check_X_y(X, y)

        self._le = LabelEncoder()
        self._le.fit(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(
                X, y, random_state
            )
            for _ in range(self.n_estimators)
        )

        self.classes_ = self.estimators_[0].classes_
        return self

    def _fit_estimator(self, X, y, random_state):
        rotation = Rotation(self.min_group_size, self.max_group_size, self.group_weight,
                            self.transformer, random_state=random_state.randint(1, 255))  # Each rotation have a different seed to create diversity
        tree = RotatedTree(self.base_estimator, rotation)

        return tree.fit(X, y)

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

        check_is_fitted(self)

        predictions = np.asarray([est.predict(X)
                                 for est in self.estimators_]).T

        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(self._le.transform(x))),
            axis=1,
            arr=predictions,
        )
        return self._le.inverse_transform(maj)

    def predict_proba(self, X, **kwargs):
        """Predict class probabilities for X.

        The probability for each instance is the mean between all classifiers in ensemble.


        Args:
            X (array-like, shape (n_samples, n_features)): The input samples.

        Returns:
            array-like, shape (n_samples, n_features): The class probabilities of the input samples. The order of the classes corresponds to that in the attribute classes_.
        """
        probas = []
        for t in self.estimators_:
            predicts = t.predict_proba(X, **kwargs)
            probas.append(predicts)
        probas = np.array(probas)
        return probas.mean(axis=0)
