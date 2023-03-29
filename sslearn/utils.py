import numpy as np
import os
import math

import pandas as pd

from statsmodels.stats.proportion import proportion_confint
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import ClassifierMixin


def safe_division(dividend, divisor, epsilon):
    if divisor == 0:
        return dividend / epsilon
    return dividend / divisor


def confidence_interval(X, hyp, y, alpha=.95):
    data = hyp.predict(X)

    successes = np.count_nonzero(data == y)
    trials = X.shape[0]
    li, hi = proportion_confint(successes, trials, alpha=1 - alpha, method="wilson")
    return li, hi


def choice_with_proportion(predictions, class_predicted, proportion, extra=0):
    n = len(predictions)
    for_each_class = {c: int(n * j) for c, j in proportion.items()}
    indices = np.zeros(0)
    for c in proportion:
        instances = class_predicted == c
        to_add = np.argsort(predictions, kind="mergesort")[instances][::-1][0:for_each_class[c] + extra]
        indices = np.concatenate((indices, to_add))

    return indices.astype(int)


def calculate_prior_probability(y):
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
        u_c[u] = float(u_c[u] / instances)
    return u_c


def is_int(x):
    """Check if x is of integer type, but not boolean"""
    # From sktime: BSD 3-Clause
    # boolean are subclasses of integers in Python, so explicitly exclude them
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def mode(y):
    """Calculate the mode of a list of values

    Parameters
    ----------
    y : array-like of shape (n_samples, n_estimators)
        array of values

    Returns
    -------
    mode: array-like of shape (n_samples,)
        array of mode of each label
    count: array-like of shape (n_samples,)
        array of count of the mode of each label
    """
    array = pd.DataFrame(np.array(y))
    mode = array.mode(axis=0).loc[0, :]
    count = array.apply(lambda x: x.value_counts().max())
    return mode.values, count.values


def check_n_jobs(n_jobs):
    """Check `n_jobs` parameter according to the scikit-learn convention.
    From sktime: BSD 3-Clause
    Parameters
    ----------
    n_jobs : int, positive or -1
        The number of jobs for parallelization.

    Returns
    -------
    n_jobs : int
        Checked number of jobs.
    """
    # scikit-learn convention
    # https://scikit-learn.org/stable/glossary.html#term-n-jobs
    if n_jobs is None:
        return 1
    elif not is_int(n_jobs):
        raise ValueError(f"`n_jobs` must be None or an integer, but found: {n_jobs}")
    elif n_jobs < 0:
        return os.cpu_count()
    else:
        return n_jobs


def calc_number_per_class(y_label):
    classes = np.unique(y_label)
    proportion = calculate_prior_probability(y_label)
    factor = 1/min(proportion.values())
    number_per_class = dict()
    for c in classes:
        number_per_class[c] = math.ceil(proportion[c] * factor)

    return number_per_class


def check_classifier(base_classifier, can_be_list=True, collection_size=None):

    if base_classifier is None:
        return DecisionTreeClassifier()
    elif can_be_list and (type(base_classifier) == list or type(base_classifier) == tuple):
        if collection_size is not None:
            if len(base_classifier) != collection_size:
                raise AttributeError(f"base_classifier is a list of classifiers, but its length ({len(base_classifier)}) is different from expected ({collection_size})")
        for i, bc in enumerate(base_classifier):
            base_classifier[i] = check_classifier(bc, False)
        return list(base_classifier)  # Transform to list
    else:
        if not isinstance(base_classifier, ClassifierMixin):
            raise AttributeError(f"base_classifier must be a ClassifierMixin, but found {type(base_classifier)}")
        return base_classifier
