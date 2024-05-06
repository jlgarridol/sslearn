import sys
import warnings
from abc import abstractmethod

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone as skclone
from sklearn.ensemble import BaggingClassifier
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array, check_random_state, resample
from sklearn.utils.validation import check_is_fitted


from sslearn.utils import check_n_jobs

from ..base import BaseEnsemble, get_dataset
from ..utils import (calc_number_per_class, calculate_prior_probability, check_classifier,
                     choice_with_proportion, confidence_interval, mode, safe_division)


class BaseCoTraining(BaseEnsemble):
    """
    Base class for CoTraining classifiers.

    Include
    -------
    1. `predict_proba` method that returns the probability of each class.
    2. `predict` method that returns the class of each instance by argmax of `predict_proba`.
    3. `score` method that returns the mean accuracy on the given test data and labels.
    """

    _estimator_type = "classifier"

    @abstractmethod
    def fit(self, X, y, **kwards):
        pass

    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        class probabilities: ndarray of shape (n_samples, n_classes)
            Array with prediction probabilities.
        """
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            columns = X.columns
        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X = check_array(X)
        if is_df:
            X = pd.DataFrame(X, columns=columns)
        if "h_" in dir(self):
            if hasattr(self.h_[0], "predict_proba"):
                ys = [
                    h.predict_proba(X[:, c] if not is_df else X.iloc[:, c])
                    for h, c in zip(self.h_, self.columns_)
                ]
                y = sum(ys) / len(ys)
            else:
                if not hasattr(self, "_one_hot_not_proba"):
                    self._one_hot_not_proba = OneHotEncoder(sparse_output=False)
                    self._one_hot_not_proba.fit(np.array(self.classes_, dtype=type(self.classes_[0])).reshape(-1, 1))
                base = np.zeros((X.shape[0], len(self.classes_)), np.float)
                for h in self.h_:
                    base += self._one_hot_not_proba.transform(h.predict(X).reshape(-1, 1))
                y = softmax(base)
            return y
        else:
            raise NotFittedError("Classifier not fitted")
        
        _estimator_type = "classifier"

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` w.r.t. `y`.
        """
        from .metrics import accuracy_score

        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


class DemocraticCoLearning(BaseCoTraining):
    """
    **Democratic Co-learning. Ensemble of classifiers of different types.**
    --------------------------------------------

    A iterative algorithm that uses a ensemble of classifiers to label instances.
    The main process is:
    1. Train each classifier with the labeled instances.
    2. While any classifier is retrained:
        1. Predict the instances from the unlabeled set.
        2. Calculate the confidence interval for each classifier for define weights.
        3. Calculate the weighted vote for each instance.
        4. Calculate the majority vote for each instance.
        5. Select the instances to label if majority vote is the same as weighted vote.
        6. Select the instances to retrain the classifier, if `only_mislabeled` is False then select all instances, else select only mislabeled instances for each classifier.
        7. Retrain the classifier with the new instances if the error rate is lower than the previous iteration.
    3. Ignore the classifiers with confidence interval lower than 0.5.
    4. Combine the probabilities of each classifier.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.
    
    
    **Example**
    -------
    ```python
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sslearn.wrapper import DemocraticCoLearning
    from sslearn.model_selection import artificial_ssl_dataset

    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, y_unlabel, _, _ = artificial_ssl_dataset(X, y, label_rate=0.1, random_state=0)
    dcl = DemocraticCoLearning(base_estimator=[DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier(n_neighbors=3)])
    dcl.fit(X, y)
    dcl.score(X_unlabel, y_unlabel)
    ``` 

    **References**
    ----------
    Y. Zhou and S. Goldman, (2004) <br>
    Democratic co-learning, <br>
    in <i>16th IEEE International Conference on Tools with Artificial Intelligence</i>,<br>
    pp. 594-602, [10.1109/ICTAI.2004.48](https://doi.org/10.1109/ICTAI.2004.48).
    """

    def __init__(
        self,
        base_estimator=[
            DecisionTreeClassifier(),
            GaussianNB(),
            KNeighborsClassifier(n_neighbors=3),
        ],
        n_estimators=None,
        expand_only_mislabeled=True,
        alpha=0.95,
        q_exp=2,
        random_state=None
    ):
        """
        Democratic Co-learning. Ensemble of classifiers of different types.

        Parameters
        ----------
        base_estimator : {ClassifierMixin, list}, optional
            An estimator object implementing fit and predict_proba or a list of ClassifierMixin, by default DecisionTreeClassifier()
        n_estimators : int, optional
            number of base_estimators to use. None if base_estimator is a list, by default None
        expand_only_mislabeled : bool, optional
            expand only mislabeled instances by itself, by default True
        alpha : float, optional
            confidence level, by default 0.95
        q_exp : int, optional
            exponent for the estimation for error rate, by default 2
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        Raises
        ------
        AttributeError
            If n_estimators is None and base_estimator is not a list
        """

        if isinstance(base_estimator, ClassifierMixin) and n_estimators is not None:
            estimators = list()
            random_available = True
            rand = check_random_state(random_state)
            if "random_state" not in dir(base_estimator):
                warnings.warn(
                    "The classifier will not be able to converge correctly, there is not enough diversity among the estimators (learners should be different).",
                    ConvergenceWarning,
                )
                random_available = False
            for i in range(n_estimators):
                estimators.append(skclone(base_estimator))
                if random_available:
                    estimators[i].random_state = rand.randint(0, 1e5)
            self.base_estimator = estimators

        elif isinstance(base_estimator, list):
            self.base_estimator = base_estimator
        else:
            raise AttributeError(
                "If `n_estimators` is None then `base_estimator` must be a `list`."
            )
        self.base_estimator = check_classifier(self.base_estimator)
        self.n_estimators = len(self.base_estimator)
        self.one_hot = OneHotEncoder(sparse_output=False)
        self.expand_only_mislabeled = expand_only_mislabeled

        self.alpha = alpha
        self.q_exp = q_exp
        self.random_state = random_state

    def __weighted_y(self, predictions, weights):
        y_complete = np.sum(
            [
                self.one_hot.transform(p.reshape(-1, 1)) * wi
                for p, wi in zip(predictions, weights)
            ],
            0,
        )
        y_zeros = np.zeros(y_complete.shape)
        y_zeros[np.arange(y_complete.shape[0]), y_complete.argmax(1)] = 1

        return self.one_hot.inverse_transform(y_zeros).flatten()

    def __calcule_last_confidences(self, X, y):
        """Calculate the confidence of each learner

        Parameters
        ----------
        X : array-like
            Set of instances
        y : array-like
            Set of classes for each instance
        """
        w = []
        w = [sum(confidence_interval(X, H, y, self.alpha)) / 2 for H in self.h_]
        self.confidences_ = w

    def fit(self, X, y, estimator_kwards=None):
        """Fit Democratic-Co classifier

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabel.
        estimator_kwards : {list, dict}, optional
            list of kwards for each estimator or kwards for all estimators, by default None

        Returns
        -------
        self : DemocraticCoLearning
            fitted classifier
        """

        X_label, y_label, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        self.classes_ = np.unique(y_label)
        self.encoder = LabelEncoder().fit(y_label)
        y_label = self.encoder.transform(y_label)

        self.one_hot.fit(y_label.reshape(-1, 1))

        L = [X_label] * self.n_estimators
        Ly = [y_label] * self.n_estimators
        # This variable prevents duplicate instances.
        L_added = [np.zeros(X_unlabel.shape[0]).astype(bool)] * self.n_estimators
        e = [0] * self.n_estimators

        if estimator_kwards is None:
            estimator_kwards = [{}] * self.n_estimators

        changed = True
        iteration = 0
        while changed:
            changed = False
            iteration_dict = {}
            iteration += 1

            for i in range(self.n_estimators):
                self.base_estimator[i].fit(L[i], Ly[i], **estimator_kwards[i])
            if X_unlabel.shape[0] == 0:
                break
            # Majority Vote
            predictions = [H.predict(X_unlabel) for H in self.base_estimator]
            majority_class = mode(np.array(predictions, dtype=predictions[0].dtype))[0]
            # majority_class = st.mode(np.array(predictions, dtype=predictions[0].dtype), axis=0, keepdims=True)[
            #     0
            # ].flatten()  # K in pseudocode

            L_ = [[]] * self.n_estimators
            Ly_ = [[]] * self.n_estimators

            # Calculate confidence interval
            conf_interval = [
                confidence_interval(
                    X_label,
                    H,
                    y_label,
                    self.alpha
                )
                for H in self.base_estimator
            ]

            weights = [(li + hi) / 2 for (li, hi) in conf_interval]
            iteration_dict["weights"] = {
                "cl" + str(i): (l, h, w)
                for i, ((l, h), w) in enumerate(zip(conf_interval, weights))
            }
            # weighted vote
            weighted_class = self.__weighted_y(predictions, weights)

            # If `weighted_class` is equal as `majority_class` then
            # the sum of classifier's weights of max voted class
            # is greater than the max of sum of classifier's weights
            # from another classes.

            candidates = weighted_class == majority_class
            candidates_bool = list()

            if not self.expand_only_mislabeled:
                all_same_list = list()
                for i in range(1, self.n_estimators):
                    all_same_list.append(predictions[i] == predictions[i - 1])
                all_same = np.logical_and(*all_same_list)
            # new_instances = []
            for i in range(self.n_estimators):

                mispredictions = predictions[i] != weighted_class
                # An instance from U are added to Li' only if:
                #   It is a misprediction for i
                #   It is a candidate (weighted_class are same majority_class)
                #   It hasn't been added yet in Li

                candidates_temp = np.logical_and(mispredictions, candidates)

                if not self.expand_only_mislabeled:
                    candidates_temp = np.logical_or(candidates_temp, all_same)

                to_add = np.logical_and(np.logical_not(L_added[i]), candidates_temp)

                candidates_bool.append(to_add)
                if is_df:
                    L_[i] = X_unlabel.iloc[to_add, :]
                else:
                    L_[i] = X_unlabel[to_add, :]
                Ly_[i] = weighted_class[to_add]

            new_conf_interval = [
                confidence_interval(L[i], H, Ly[i], self.alpha)
                for i, H in enumerate(self.base_estimator)
            ]
            e_factor = 1 - sum([l_ for l_, _ in new_conf_interval]) / self.n_estimators
            for i, _ in enumerate(self.base_estimator):
                if len(L_[i]) > 0:

                    qi = len(L[i]) * ((1 - 2 * (e[i] / len(L[i]))) ** 2)
                    e_i = e_factor * len(L_[i])
                    # |Li|+|L'i| == |Li U L'i| because of to_add
                    q_i = (len(L[i]) + len(L_[i])) * (
                        1 - 2 * (e[i] + e_i) / (len(L[i]) + len(L_[i]))
                    ) ** self.q_exp
                    if q_i <= qi:
                        continue
                    L_added[i] = np.logical_or(L_added[i], candidates_bool[i])
                    if is_df:
                        L[i] = pd.concat([L[i], L_[i]])
                    else:
                        L[i] = np.concatenate((L[i], np.array(L_[i])))
                    Ly[i] = np.concatenate((Ly[i], np.array(Ly_[i])))

                    e[i] = e[i] + e_i
                    changed = True

        self.h_ = self.base_estimator
        self.__calcule_last_confidences(X_label, y_label)

        # Ignore hypothesis
        self.h_ = [H for w, H in zip(self.confidences_, self.h_) if w > 0.5]
        self.confidences_ = [w for w in self.confidences_ if w > 0.5]

        self.columns_ = [list(range(X.shape[1]))] * self.n_estimators

        return self

    def __combine_probabilities(self, X):

        n_instances = X.shape[0]  # uppercase X as it will be an np.array
        sizes = np.zeros((n_instances, len(self.classes_)), dtype=int)
        C = np.zeros((n_instances, len(self.classes_)), dtype=float)
        Cavg = np.zeros((n_instances, len(self.classes_)), dtype=float)

        for w, H in zip(self.confidences_, self.h_):
            cj = H.predict(X)
            factor = self.one_hot.transform(cj.reshape(-1, 1)).astype(int)
            C += w * factor
            sizes += factor

        Cavg[sizes == 0] = 0.5  # «voting power» of 0.5 for small groups
        ne = (sizes != 0)  # non empty groups
        Cavg[ne] = (sizes[ne] + 0.5) / (sizes[ne] + 1) * C[ne] / sizes[ne]

        return softmax(Cavg, axis=1)

    def predict_proba(self, X):
        """Predict probability for each possible outcome.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        class probabilities: ndarray of shape (n_samples, n_classes)
            Array with prediction probabilities.
        """
        if "h_" in dir(self):
            if len(X) == 1:
                X = [X]
            return self.__combine_probabilities(X)
        else:
            raise NotFittedError("Classifier not fitted")


class CoTraining(BaseCoTraining):
    """
    **CoTraining classifier. Multi-view learning algorithm that uses two classifiers to label instances.**
    --------------------------------------------

    The main process is:
    1. Train each classifier with the labeled instances and their respective view.
    2. While max iterations is not reached or any instance is unlabeled:
        1. Predict the instances from the unlabeled set.
        2. Select the instances that have the same prediction and the predictions are above the threshold.
        3. Label the instances with the highest probability, keeping the balance of the classes.
        4. Retrain the classifier with the new instances.
    3. Combine the probabilities of each classifier.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **Example**
    -------
    ```python
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    from sslearn.wrapper import CoTraining
    from sslearn.model_selection import artificial_ssl_dataset

    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, y_unlabel, _, _ = artificial_ssl_dataset(X, y, label_rate=0.1, random_state=0)
    cotraining = CoTraining(DecisionTreeClassifier())
    X1 = X[:, [0, 1]]
    X2 = X[:, [2, 3]]
    cotraining.fit(X1, y, X2) 
    # or
    cotraining.fit(X, y, features=[[0, 1], [2, 3]])
    # or
    cotraining = CoTraining(DecisionTreeClassifier(), force_second_view=False)
    cotraining.fit(X, y)
    ``` 

    **References**
    ----------
    Avrim Blum and Tom Mitchell. (1998).<br>
    Combining labeled and unlabeled data with co-training<br>
    in <i>Proceedings of the eleventh annual conference on Computational learning theory (COLT' 98)</i>.<br>
    Association for Computing Machinery, New York, NY, USA, 92-100.<br>
    [10.1145/279943.279962](https://doi.org/10.1145/279943.279962)

    Han, Xian-Hua, Yen-wei Chen, and Xiang Ruan. (2011). <br>
    Multi-Class Co-Training Learning for Object and Scene Recognition,<br>
    pp. 67-70 in. Nara, Japan. <br>
    [http://www.mva-org.jp/Proceedings/2011CD/papers/04-08.pdf](http://www.mva-org.jp/Proceedings/2011CD/papers/04-08.pdf)<br>
    """

    def __init__(
        self,
        base_estimator=DecisionTreeClassifier(),
        second_base_estimator=None,
        max_iterations=30,
        poolsize=75,
        threshold=0.5,
        force_second_view=True,
        random_state=None
    ):
        """
        Create a CoTraining classifier. 
        Multi-view learning algorithm that uses two classifiers to label instances.

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
        threshold : float, optional
            The threshold for label instances, by default 0.5
        force_second_view : bool, optional
            The second classifier needs a different view of the data. If False then a second view will be same as the first, by default True
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None

        """
        self.base_estimator = check_classifier(base_estimator, False)
        if second_base_estimator is not None:
            second_base_estimator = check_classifier(second_base_estimator, False)
        self.second_base_estimator = second_base_estimator
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.threshold = threshold
        self.force_second_view = force_second_view
        self.random_state = random_state

    def fit(self, X, y, X2=None, features: list = None, number_per_class: dict = None, **kwards):
        """
        Build a CoTraining classifier from the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabeled.
        X2 : {array-like, sparse matrix} of shape (n_samples, n_features), optional
            Array representing the data from another view, not compatible with `features`, by default None
        features : {list, tuple}, optional
            list or tuple of two arrays with `feature` index for each subspace view, not compatible with `X2`, by default None
        number_per_class : {dict}, optional
            dict of class name:integer with the max ammount of instances to label in this class in each iteration, by default None

        Returns
        -------
        self: CoTraining
            Fitted estimator.
        """
        rs = check_random_state(self.random_state)

        X_label, y_label, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        if X2 is not None:
            X2_label, _, X2_unlabel = get_dataset(X2, y)
        elif features is not None:
            if is_df:
                X2_label = X_label.iloc[:, features[1]]
                X2_unlabel = X_unlabel.iloc[:, features[1]]
                X_label = X_label.iloc[:, features[0]]
                X_unlabel = X_unlabel.iloc[:, features[0]]
            else:
                X2_label = X_label[:, features[1]]
                X2_unlabel = X_unlabel[:, features[1]]
                X_label = X_label[:, features[0]]
                X_unlabel = X_unlabel[:, features[0]]
            self.columns_ = features
        elif self.force_second_view:
            raise AttributeError("Either X2 or features must be defined. CoTraining need another view to train the second classifier")
        else:
            self.columns_ = [list(range(X.shape[1]))] * 2
            X2_label = X_label.copy()
            X2_unlabel = X_unlabel.copy()

        if is_df and X2_label is not None and not isinstance(X2_label, pd.DataFrame):
            raise AttributeError("X and X2 must be both pandas DataFrame or numpy arrays")

        self.h = [
            skclone(self.base_estimator),
            skclone(self.base_estimator) if self.second_base_estimator is None else skclone(self.second_base_estimator)
        ]
        assert (
            X2 is None or features is None
        ), "The list of features and X2 cannot be defined at the same time"

        self.classes_ = np.unique(y_label)
        if number_per_class is None:
            number_per_class = calc_number_per_class(y_label)

        if X_unlabel.shape[0] < self.poolsize:
            warnings.warn(f"Poolsize ({self.poolsize}) is bigger than U ({X_unlabel.shape[0]})")

        permutation = rs.permutation(len(X_unlabel))

        self.h[0].fit(X_label, y_label)
        self.h[1].fit(X2_label, y_label)

        it = 0
        while it < self.max_iterations and any(permutation):
            it += 1

            get_index = permutation[:self.poolsize]
            y1_prob = self.h[0].predict_proba(X_unlabel[get_index] if not is_df else X_unlabel.iloc[get_index, :])
            y2_prob = self.h[1].predict_proba(X2_unlabel[get_index] if not is_df else X2_unlabel.iloc[get_index, :])

            predictions1 = np.max(y1_prob, axis=1)
            class_predicted1 = np.argmax(y1_prob, axis=1)

            predictions2 = np.max(y2_prob, axis=1)
            class_predicted2 = np.argmax(y2_prob, axis=1)

            # If two classifier select same instance and bring different predictions then the instance is not labeled
            candidates1 = predictions1 > self.threshold
            candidates2 = predictions2 > self.threshold
            aggreement = class_predicted1 == class_predicted2

            full_candidates = candidates1 ^ candidates2
            medium_candidates = candidates1 & candidates2 & aggreement
            true_candidates1 = full_candidates & candidates1
            true_candidates2 = full_candidates & candidates2

            # Fill probas and candidate classes.
            y_probas = np.zeros(predictions1.shape, dtype=predictions1.dtype)
            y_class = class_predicted1.copy()

            temp_probas1 = predictions1[true_candidates1]
            temp_probas2 = predictions2[true_candidates2]
            temp_probasB = (predictions1[medium_candidates]+predictions2[medium_candidates])/2

            temp_classes2 = class_predicted2[true_candidates2]

            y_probas[true_candidates1] = temp_probas1
            y_probas[true_candidates2] = temp_probas2
            y_probas[medium_candidates] = temp_probasB
            y_class[true_candidates2] = temp_classes2

            # Select the best candidates
            final_instances = list()
            best_candidates = np.argsort(y_probas, kind="mergesort")[::-1]
            for c in self.classes_:
                final_instances += list(best_candidates[y_class[best_candidates] == c])[:number_per_class[c]]

            # Fill the new labeled instances
            pseudoy = y_class[final_instances]
            y_label = np.append(y_label, pseudoy)

            index = permutation[0: self.poolsize][final_instances]
            if is_df:
                X_label = pd.concat([X_label, X_unlabel.iloc[index, :]])
                X2_label = pd.concat([X2_label, X2_unlabel.iloc[index, :]])
            else:
                X_label = np.append(X_label, X_unlabel[index], axis=0)
                X2_label = np.append(X2_label, X2_unlabel[index], axis=0)

            permutation = permutation[list(map(lambda x: x not in index, permutation))]

            # Poolsize increments in order double of max instances candidates:
            self.poolsize += sum(number_per_class.values()) * 2

            self.h[0].fit(X_label, y_label)
            self.h[1].fit(X2_label, y_label)

        self.h_ = self.h

        return self

    def predict_proba(self, X, X2=None, **kwards):
        """Predict probability for each possible outcome.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        X2 : {array-like, sparse matrix} of shape (n_samples, n_features), optional
            Array representing the data from another view, by default None
        Returns
        -------
        class probabilities: ndarray of shape (n_samples, n_classes)
            Array with prediction probabilities.
        """
        if "columns_" in dir(self):
            return super().predict_proba(X, **kwards)
        elif "h_" in dir(self):
            ys = []
            ys.append(self.h_[0].predict_proba(X, **kwards))
            ys.append(self.h_[1].predict_proba(X2, **kwards))
            y = sum(ys) / len(ys)
            return y
        else:
            raise NotFittedError("Classifier not fitted")

    def predict(self, X, X2=None, **kwards):
        """Predict the classes of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        X2 : {array-like, sparse matrix} of shape (n_samples, n_features), optional
            Array representing the data from another view, by default None

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array with predicted labels.
        """
        if "columns_" in dir(self):
            result = super().predict(X, **kwards)
        else:
            predicted_probabilitiy = self.predict_proba(X, X2, **kwards)
            result = self.classes_.take(
                (np.argmax(predicted_probabilitiy, axis=1)), axis=0
            )
        return result

    def score(self, X, y, sample_weight=None, **kwards):
        """
        Return the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for `X`.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        X2 : {array-like, sparse matrix} of shape (n_samples, n_features), optional
            Array representing the data from another view, by default None
        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        if "X2" in kwards:
            return accuracy_score(y, self.predict(X, kwards["X2"]), sample_weight=sample_weight)
        else:
            return super().score(X, y, sample_weight=sample_weight)


class Rasco(BaseCoTraining):
    """
    **Co-Training based on random subspaces**
    --------------------------------------------

    Generate a set of random subspaces and train a classifier for each subspace.

    The main process is:
    1. Generate a set of random subspaces.
    2. Train a classifier for each subspace.
    3. While max iterations is not reached or any instance is unlabeled:
        1. Predict the instances from the unlabeled set for each classifier.
        2. Calculate the average of the predictions.
        3. Select the instances with the highest probability.
        4. Label the instances with the highest probability, keeping the balance of the classes.
        5. Retrain the classifier with the new instances.
    4. Combine the probabilities of each classifier.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **Example**
    -------
    ```python
    from sklearn.datasets import load_iris
    from sslearn.wrapper import Rasco
    from sslearn.model_selection import artificial_ssl_dataset

    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, y_unlabel, _, _ = artificial_ssl_dataset(X, y, label_rate=0.1, random_state=0)
    rasco = Rasco()
    rasco.fit(X, y)
    rasco.score(X_unlabel, y_unlabel) 
    ```    

    **References**
    ----------
    Wang, J., Luo, S. W., & Zeng, X. H. (2008).<br>
    A random subspace method for co-training,<br>
    in <i>2008 IEEE International Joint Conference on Neural Networks</i><br>
    IEEE World Congress on Computational Intelligence<br>
    (pp. 195-200). IEEE. [10.1109/IJCNN.2008.4633789](https://doi.org/10.1109/IJCNN.2008.4633789)
    """


    def __init__(
        self,
        base_estimator=DecisionTreeClassifier(),
        max_iterations=10,
        n_estimators=30,
        subspace_size=None,
        random_state=None,
        n_jobs=None,
    ):
        """
        Co-Training based on random subspaces

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        max_iterations : int, optional
            Maximum number of iterations allowed. Should be greater than or equal to 0.
            If is -1 then will be infinite iterations until U be empty, by default 10
        n_estimators : int, optional
            The number of base estimators in the ensemble., by default 30
        subspace_size : int, optional
            The number of features for each subspace. If it is None will be the half of the features size., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = check_classifier(base_estimator, True, n_estimators)  # C in paper
        self.max_iterations = max_iterations  # J in paper
        self.n_estimators = n_estimators  # K in paper
        self.subspace_size = subspace_size  # m in paper
        self.n_jobs = check_n_jobs(n_jobs)

        self.random_state = random_state

    def _generate_random_subspaces(self, X, y=None, random_state=None):
        """Generate the random subspaces

        Parameters
        ----------
        X : array like
            Labeled dataset
        y : array like, optional
            Target for each X, not needed on Rasco, by default None

        Returns
        -------
        subspaces : list
            List of index of features
        """
        random_state = check_random_state(random_state)
        features = list(range(X.shape[1]))
        idxs = []
        for _ in range(self.n_estimators):
            idxs.append(random_state.permutation(features)[: self.subspace_size])
        return idxs

    def _fit_estimator(self, X, y, i, **kwards):
        estimator = self.base_estimator
        if type(self.base_estimator) == list:
            estimator = skclone(self.base_estimator[i])
        return skclone(estimator).fit(X, y, **kwards)

    def fit(self, X, y, **kwards):
        """Build a Rasco classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabel.

        Returns
        -------
        self: Rasco
            Fitted estimator.
        """
        X_label, y_label, X_unlabel = get_dataset(X, y)
        self.classes_ = np.unique(y_label)

        is_df = isinstance(X_label, pd.DataFrame)

        random_state = check_random_state(self.random_state)

        self.classes_ = np.unique(y_label)
        number_per_class = calc_number_per_class(y_label)

        if self.subspace_size is None:
            self.subspace_size = int(X.shape[1] / 2)
        idxs = self._generate_random_subspaces(X_label, y_label, random_state)

        cfs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_estimator)(X_label[:, idxs[i]] if not is_df else X_label.iloc[:, idxs[i]], y_label, i, **kwards)
            for i in range(self.n_estimators)
        )

        it = 0
        while True:
            if (self.max_iterations != -1 and it >= self.max_iterations) or len(
                X_unlabel
            ) == 0:
                break

            raw_predicions = []
            for i in range(self.n_estimators):
                rp = cfs[i].predict_proba(X_unlabel[:, idxs[i]] if not is_df else X_unlabel.iloc[:, idxs[i]])
                raw_predicions.append(rp)
            raw_predicions = sum(raw_predicions) / self.n_estimators
            predictions = np.max(raw_predicions, axis=1)
            class_predicted = np.argmax(raw_predicions, axis=1)
            pseudoy = self.classes_.take(class_predicted, axis=0)

            final_instances = list()
            best_candidates = np.argsort(predictions, kind="mergesort")[::-1]
            for c in self.classes_:
                final_instances += list(best_candidates[pseudoy[best_candidates] == c])[:number_per_class[c]]

            Lj = X_unlabel[final_instances] if not is_df else X_unlabel.iloc[final_instances]
            yj = pseudoy[final_instances]

            X_label = np.append(X_label, Lj, axis=0) if not is_df else pd.concat([X_label, Lj])
            y_label = np.append(y_label, yj)
            X_unlabel = np.delete(X_unlabel, final_instances, axis=0) if not is_df else X_unlabel.drop(index=X_unlabel.index[final_instances])

            cfs = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_estimator)(X_label[:, idxs[i]] if not is_df else X_label.iloc[:, idxs[i]], y_label, i, **kwards)
                for i in range(self.n_estimators)
            )

            it += 1

        self.h_ = cfs
        self.columns_ = idxs

        return self


class RelRasco(Rasco):
    """
    **Co-Training based on relevant random subspaces**
    --------------------------------------------

    Is a variation of `sslearn.wrapper.Rasco` that uses the mutual information of each feature to select the random subspaces.
    The process of training is the same as Rasco.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **Example**
    -------
    ```python
    from sklearn.datasets import load_iris
    from sslearn.wrapper import RelRasco
    from sslearn.model_selection import artificial_ssl_dataset

    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, y_unlabel, _, _ = artificial_ssl_dataset(X, y, label_rate=0.1, random_state=0)
    relrasco = RelRasco()
    relrasco.fit(X, y)
    relrasco.score(X_unlabel, y_unlabel)
    ```

    **References**
    ----------
    Yaslan, Y., & Cataltepe, Z. (2010).<br>
    Co-training with relevant random subspaces.<br>
    <i>Neurocomputing</i>, 73(10-12), 1652-1661.<br>
    [10.1016/j.neucom.2010.01.018](https://doi.org/10.1016/j.neucom.2010.01.018)
    """

    def __init__(
        self,
        base_estimator=DecisionTreeClassifier(),
        max_iterations=10,
        n_estimators=30,
        subspace_size=None,
        random_state=None,
        n_jobs=None,
    ):
        """
        Co-Training with relevant random subspaces

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        max_iterations : int, optional
            Maximum number of iterations allowed. Should be greater than or equal to 0.
            If is -1 then will be infinite iterations until U be empty, by default 10
        n_estimators : int, optional
            The number of base estimators in the ensemble., by default 30
        subspace_size : int, optional
            The number of features for each subspace. If it is None will be the half of the features size., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        n_jobs : int, optional
            The number of jobs to run in parallel. -1 means using all processors., by default None

        """
        super().__init__(
            base_estimator,
            max_iterations,
            n_estimators,
            subspace_size,
            random_state,
            n_jobs,
        )

    def _generate_random_subspaces(self, X, y, random_state=None):
        """Generate the relevant random subspcaes

        Parameters
        ----------
        X : array like
            Labeled dataset
        y : array like, optional
            Target for each X, only needed on Rel-Rasco, by default None

        Returns
        -------
        subspaces: list
            List of index of features
        """
        random_state = check_random_state(random_state)
        relevance = mutual_info_classif(X, y, random_state=random_state)
        idxs = []
        for _ in range(self.n_estimators):
            subspace = []
            for __ in range(self.subspace_size):
                f1 = random_state.randint(0, X.shape[1])
                f2 = random_state.randint(0, X.shape[1])
                if relevance[f1] > relevance[f2]:
                    subspace.append(f1)
                else:
                    subspace.append(f2)
            idxs.append(subspace)
        return idxs


# Done and tested
class CoTrainingByCommittee(BaseCoTraining):
    """
    **Co-Training by Committee classifier.**
    --------------------------------------------

    Create a committee trained by co-training based on the diversity of the classifiers

    The main process is:
    1. Train a committee of classifiers.
    2. Create a pool of unlabeled instances.
    3. While max iterations is not reached or any instance is unlabeled:
        1. Predict the instances from the unlabeled set.
        2. Select the instances with the highest probability.
        3. Label the instances with the highest probability, keeping the balance of the classes but ensuring that at least n instances of each class are added.
        4. Retrain the classifier with the new instances.
    4. Combine the probabilities of each classifier.

    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **Example**
    -------
    ```python
    from sklearn.datasets import load_iris
    from sslearn.wrapper import CoTrainingByCommittee
    from sslearn.model_selection import artificial_ssl_dataset

    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, y_unlabel, _, _ = artificial_ssl_dataset(X, y, label_rate=0.1, random_state=0)
    cotraining = CoTrainingByCommittee()
    cotraining.fit(X, y)
    cotraining.score(X_unlabel, y_unlabel)
    ```

    **References**
    ----------
    M. F. A. Hady and F. Schwenker,<br>
    Co-training by Committee: A New Semi-supervised Learning Framework,<br>
    in <i>2008 IEEE International Conference on Data Mining Workshops</i>,<br>
    Pisa, 2008, pp. 563-572,  [10.1109/ICDMW.2008.27](https://doi.org/10.1109/ICDMW.2008.27)
    """

    def __init__(
        self,
        ensemble_estimator=BaggingClassifier(),
        max_iterations=100,
        poolsize=100,
        min_instances_for_class=3,
        random_state=None,
    ):
        """
        Create a committee trained by cotraining based on
        the diversity of classifiers.

        Parameters
        ----------
        ensemble_estimator : ClassifierMixin, optional
            ensemble method, works without a ensemble as
            self training with pool, by default BaggingClassifier().
        max_iterations : int, optional
            number of iterations of training, -1 if no max iterations, by default 100
        poolsize : int, optional
            max number of unlabeled instances candidates to pseudolabel, by default 100
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None


        """
        self.ensemble_estimator = check_classifier(ensemble_estimator, False)
        self.max_iterations = max_iterations
        self.poolsize = poolsize
        self.random_state = random_state
        self.min_instances_for_class = min_instances_for_class

    def fit(self, X, y, **kwards):
        """Build a CoTrainingByCommittee classifier from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabel.
        Returns
        -------
        self : CoTrainingByCommittee
            Fitted estimator.
        """
        self.ensemble_estimator = skclone(self.ensemble_estimator)
        random_state = check_random_state(self.random_state)

        X_label, y_prev, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        self.label_encoder_ = LabelEncoder()
        y_label = self.label_encoder_.fit_transform(y_prev)

        self.classes_ = self.label_encoder_.classes_

        prior = calculate_prior_probability(y_label)
        permutation = random_state.permutation(len(X_unlabel))

        self.ensemble_estimator.fit(X_label, y_label, **kwards)

        if X_unlabel.shape[0] == 0:
            return self

        for _ in range(self.max_iterations):
            if len(permutation) == 0:
                break
            raw_predictions = self.ensemble_estimator.predict_proba(
                X_unlabel[permutation[0: self.poolsize]] if not is_df else X_unlabel.iloc[permutation[0: self.poolsize]]
            )

            predictions = np.max(raw_predictions, axis=1)
            class_predicted = np.argmax(raw_predictions, axis=1)

            added = np.zeros(predictions.shape, dtype=bool)
            # First the n (or less) most confidence instances will be selected
            for c in self.ensemble_estimator.classes_:
                condition = class_predicted == c

                candidates = predictions[condition]
                candidates_bool = np.zeros(predictions.shape, dtype=bool)
                candidates_sub_set = candidates_bool[condition]

                instances_index_selected = candidates.argsort(kind="mergesort")[
                    -self.min_instances_for_class:
                ]

                candidates_sub_set[instances_index_selected] = True
                candidates_bool[condition] += candidates_sub_set

                added[candidates_bool] = True

            # Bajo esta interpretación se garantiza que al menos existen n elemento de cada clase por iteración
            # Pero si se añaden ya en el proceso de proporción no se duplica.

            # Con esta otra interpretación ignora las n primeras instancias de cada clase
            to_label = choice_with_proportion(
                predictions, class_predicted, prior, extra=self.min_instances_for_class
            )
            added[to_label] = True

            index = permutation[0: self.poolsize][added]
            X_label = np.append(X_label, X_unlabel[index], axis=0) if not is_df else pd.concat(
                [X_label, X_unlabel.iloc[index, :]]
            )
            pseudoy = class_predicted[added]

            y_label = np.append(y_label, pseudoy)
            permutation = permutation[list(map(lambda x: x not in index, permutation))]

            self.ensemble_estimator.fit(X_label, y_label, **kwards)

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
        y : array-like of shape (n_samples,)
            The predicted classes
        """
        check_is_fitted(self.ensemble_estimator)
        return self.label_encoder_.inverse_transform(self.ensemble_estimator.predict(X))

    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.
        The predicted class probability depends on the ensemble estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray of shape (n_samples, n_classes) or list of n_outputs such arrays if n_outputs > 1
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
        try:
            y = self.label_encoder_.transform(y)
        except ValueError:
            if "le_dict_" not in dir(self):
                self.le_dict_ = dict(
                    zip(
                        self.label_encoder_.classes_,
                        self.label_encoder_.transform(self.label_encoder_.classes_),
                    )
                )
            y = np.array(list(map(lambda x: self.le_dict_.get(x, -1), y)), dtype=y.dtype)

        return self.ensemble_estimator.score(X, y, sample_weight)

# Done and tested
class CoForest(BaseCoTraining):
    """
    **CoForest classifier. Random Forest co-training**
    ----------------------------
    
    Ensemble method for CoTraining based on Random Forest.

    The main process is:
    1. Train a committee of classifiers using bootstrap.
    2. While any base classifier is retrained:
        1. Predict the instances from the unlabeled set.
        2. Select the instances with the highest probability.
        3. Label the instances with the highest probability
        4. Add the instances to the labeled set only if the error is not bigger than the previous error.
        5. Retrain the classifier with the new instances.
    3. Combine the probabilities of each classifier.


    **Methods**
    -------
    - `fit`: Fit the model with the labeled instances.
    - `predict` : Predict the class for each instance.
    - `predict_proba`: Predict the probability for each class.
    - `score`: Return the mean accuracy on the given test data and labels.

    **Example**
    -------
    ```python
    from sklearn.datasets import load_iris
    from sslearn.wrapper import CoForest
    from sslearn.model_selection import artificial_ssl_dataset

    X, y = load_iris(return_X_y=True)
    X, y, X_unlabel, y_unlabel, _, _ = artificial_ssl_dataset(X, y, label_rate=0.1, random_state=0)
    coforest = CoForest()
    coforest.fit(X, y)
    coforest.score(X_unlabel, y_unlabel)
    ```

    **References**
    ----------
    Li, M., & Zhou, Z.-H. (2007).<br>
    Improve Computer-Aided Diagnosis With Machine Learning Techniques Using Undiagnosed Samples.<br>
    <i>IEEE Transactions on Systems, Man, and Cybernetics - Part A: Systems and Humans</i>,<br>
    37(6), 1088-1098. [10.1109/tsmca.2007.904745](https://doi.org/10.1109/tsmca.2007.904745)
    """

    def __init__(self, base_estimator=DecisionTreeClassifier(), n_estimators=7, threshold=0.75, bootstrap=True, n_jobs=None, random_state=None, version="1.0.3"):
        """
        Generate a CoForest classifier.
        A SSL Random Forest adaption for CoTraining. 

        Parameters
        ----------
        base_estimator : ClassifierMixin, optional
            An estimator object implementing fit and predict_proba, by default DecisionTreeClassifier()
        n_estimators : int, optional
            The number of base estimators in the ensemble., by default 7
        threshold : float, optional
            The decision threshold. Should be in [0, 1)., by default 0.5
        n_jobs : int, optional
            The number of jobs to run in parallel for both fit and predict., by default None
        bootstrap : bool, optional
            Whether bootstrap samples are used when building estimators., by default True
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        **kwards : dict, optional
            Additional parameters to be passed to base_estimator, by default None.
        """
        self.base_estimator = check_classifier(base_estimator, collection_size=n_estimators)
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.bootstrap = bootstrap
        self._epsilon = sys.float_info.epsilon
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.version = version
        if self.version == "1.0.2":
            warnings.warn("The version 1.0.2 is deprecated. Please use the version 1.0.3", DeprecationWarning)

    def __bootstraping(self, X, y, r_state):
        # It is necessary to bootstrap the data
        if self.bootstrap and self.version == "1.0.3":
            is_df = isinstance(X, pd.DataFrame)
            columns = None
            if is_df:
                columns = X.columns
                X = X.to_numpy()
            y = y.copy()
            # Get a reprentation of each class
            classes = np.unique(y)
            # Choose at least one sample from each class
            X_label, y_label = [], []
            for c in classes:
                index = np.where(y == c)[0]
                # Choose one sample from each class
                X_label.append(X[index[0], :])
                y_label.append(y[index[0]])
                # Remove the sample from the original data
                X = np.delete(X, index[0], axis=0)
                y = np.delete(y, index[0], axis=0)
            X, y = resample(X, y, random_state=r_state)
            X = np.concatenate((X, np.array(X_label)), axis=0)
            y = np.concatenate((y, np.array(y_label)), axis=0)
            if is_df:
                X = pd.DataFrame(X, columns=columns)
        return X, y

    def __estimate_error(self, hypothesis, X, y, index):
        if self.version == "1.0.3":
            concomitants = [h for i, h in enumerate(self.hypotheses) if i != index]
            predicted = [h.predict(X) for h in concomitants]
            predicted = np.array(predicted, dtype=y.dtype)
            # Get the majority vote
            predicted, _ = mode(predicted)
            # predicted, _ = st.mode(predicted, axis=1)
            # Get the error rate
            return 1 - accuracy_score(y, predicted)
        else:
            probas = hypothesis.predict_proba(X)
            ei_t = 0
            classes = list(hypothesis.classes_)
            for j in range(y.shape[0]):
                true_y = y[j]
                true_y_index = classes.index(true_y)
                ei_t += 1 - probas[j, true_y_index]
            if ei_t == 0:
                ei_t = self._epsilon
            return ei_t

    def __confidence(self, h_index, X):
        concomitants = [h for i, h in enumerate(self.hypotheses) if i != h_index]

        predicted = [h.predict(X) for h in concomitants]
        predicted = np.array(predicted, dtype=predicted[0].dtype)
        # Get the majority vote and the number of votes
        _, counts = mode(predicted)
        # _, counts = st.mode(predicted, axis=1)
        confidences = counts / len(concomitants)
        return confidences

    def _fit_estimator(self, X, y, i, beginning=False, **kwards):
        estimator = self.base_estimator
        if type(self.base_estimator) == list:
            estimator = skclone(self.hypotheses[i])

        if "random_state" in estimator.get_params():
            r_state = estimator.random_state
        else:
            r_state = self.random_state
            if r_state is None:
                r_state = np.random.randint(0, 1000)
            r_state += i
        # Only in the beginning
        if beginning:
            X, y = self.__bootstraping(X, y, r_state)

        return skclone(estimator).fit(X, y, **kwards)

    def fit(self, X, y, **kwards):
        """Build a CoForest classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels), -1 if unlabel.

        Returns
        -------
        self: CoForest
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        n_jobs = check_n_jobs(self.n_jobs)

        X_label, y_label, X_unlabel = get_dataset(X, y)

        is_df = isinstance(X_label, pd.DataFrame)

        self.classes_ = np.unique(y_label)

        self.hypotheses = []
        errors = []
        weights = []
        for i in range(self.n_estimators):
            self.hypotheses.append(skclone(self.base_estimator if type(self.base_estimator) is not list else self.base_estimator[i]))
            if "random_state" in dir(self.hypotheses[-1]):
                self.hypotheses[-1].set_params(random_state=random_state.randint(0, 2147483647))
            errors.append(0.5)

        self.hypotheses = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_estimator)(X_label, y_label, i, beginning=True, **kwards)
            for i in range(self.n_estimators)
        )

        for i in range(self.n_estimators):
            # The paper stablishes that the weight of each hypothesis is 0,
            # but it is not possible to do that because it will be impossible increase the training set
            if self.version == "1.0.2":
                weights.append(np.max(self.hypotheses[i].predict_proba(X_label), axis=1).sum())  # Version 1.0.2
            else:
                weights.append(self.__confidence(i, X_label).sum())

        changing = True if X_unlabel.shape[0] > 0 else False
        while changing:
            changing = False
            for i in range(self.n_estimators):
                hi, ei, wi = self.hypotheses[i], errors[i], weights[i]

                ei_t = self.__estimate_error(hi, X_label, y_label, i)

                if ei_t < ei:
                    random_index_subsample = list(range(X_unlabel.shape[0]))
                    random_index_subsample = random_state.permutation(
                        random_index_subsample
                    )
                    cond = random_index_subsample[0:int(safe_division(ei * wi, ei_t, self._epsilon))]
                    if is_df:
                        Ui_t = X_unlabel.iloc[cond, :]
                    else:
                        Ui_t = X_unlabel[cond, :]

                    raw_predictions = hi.predict_proba(Ui_t)
                    predictions = np.max(raw_predictions, axis=1)
                    class_predicted = self.classes_.take(np.argmax(raw_predictions, axis=1), axis=0)

                    to_label = predictions > self.threshold
                    wi_t = predictions[to_label].sum()

                    if ei_t * wi_t < ei * wi:
                        changing = True
                        if is_df:
                            x_temp = pd.concat([X_label, Ui_t.iloc[to_label, :]])
                        else:
                            x_temp = np.concatenate((X_label, Ui_t[to_label]))
                        y_temp = np.concatenate((y_label, class_predicted[to_label]))
                        hi.fit(
                            x_temp,
                            y_temp,
                            **kwards
                        )

                    errors[i] = ei_t
                    weights[i] = wi_t

        self.h_ = self.hypotheses
        self.columns_ = [list(range(X.shape[1]))] * self.n_estimators

        return self
