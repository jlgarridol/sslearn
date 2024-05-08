"""Summary of module `sslearn.restricted`:

This module contains classes to train a classifier using the restricted set classification approach.

## Classes

[WhoIsWhoClassifier](#WhoIsWhoClassifier):
> Who is Who Classifier

## Functions

[conflict_rate](#conflict_rate): 
> Compute the conflict rate of a prediction, given a set of restrictions.

[combine_predictions](#combine_predictions): 
> Combine the predictions of a group of instances to keep the restrictions.

[feature_fusion](#feature_fusion):
> Restricted Set Classification for the instances with pairwise constraints. Combine all instances that have the must-link constraint with the average of their features.

[probability_fusion](#probability_fusion):
> Restricted Set Classification for the instances with pairwise constraints. The class probability for each instance is defined as the mean of the probabilities reported by the classifier according to the must-link constraint.


"""

import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, BaseEstimator
from scipy.optimize import linear_sum_assignment
import warnings
import pandas as pd

__all__ = ["conflict_rate", "combine_predictions", "feature_fusion", "probability_fusion", "WhoIsWhoClassifier"]
 
    
def feature_fusion(classifier, X, must_link, cannot_link):
    """
    Restricted Set Classification for the instances with pairwise constraints. 
    Combine all instances that have the must-link constraint with the average of their features.    

    Parameters
    ----------
    classifier : ClassifierMixin with predict_proba method
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Array representing the data.
    must_link : dict of {int: list of int}
        Dictionary with the must links, where the key is the instance and the value is a list of instances that must have the same label.
    cannot_link : dict of {int: list of int}
        Dictionary with the cannot links, where the value is a list of instances that cannot have the same label.

    Returns
    ----------
    y : ndarray of shape (n_samples,)
        Array with predicted labels.

    Examples
    ----------
    ```python
    from sslearn.restricted import feature_fusion
    from sklearn.bayes import GaussianNB
    import pandas as pd

    dataset = pd.read_csv("dataset.csv")

    must_link = pd.read_csv("must_link.csv", index_col=0).to_dict(orient='index')
    # must_link = {0: [0, 2], 1: [1, 3]} -> 
    # instances 0 and 2 must have the same label, 
    # and instances 1 and 3 must have the same label

    cannot_link = pd.read_csv("cannot_link.csv", index_col=0).to_dict(orient='index')
    # cannot_link = {0: [0, 1], 1: [2, 3]} ->
    # instances 0 and 1 cannot have the same label, 
    # and instances 2 and 3 cannot have the same label

    X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
    X_label = X[y != y.dtype.type(-1)]
    y_label = y[y != y.dtype.type(-1)]
    X_unlabel = X[y == y.dtype.type(-1)]

    classifier = GaussianNB()
    classifier.fit(X_label, y_label)

    y_pred = feature_fusion(classifier, X_unlabel, must_link, cannot_link)
    ```

    References
    ----------
    L.I. Kuncheva, J.L. Garrido-Labrador, I. Ramos-Pérez, S.L. Hennessey, J.J. Rodríguez (2024).<br>
    Semi-supervised classification with pairwise constraints: A case study on animal identification from video.<br>
    <i>Information Fusion,</i><br> 
    104, 102188, [10.1016/j.inffus.2023.102188](https://doi.org/10.1016/j.inffus.2023.102188)
    """
    
    X_combined = __combine_features(X, must_link)
    y_pred_proba = classifier.predict_proba(X_combined)

    return __restricted_set_classification(y_pred_proba, cannot_link, classifier.classes_)


def probability_fusion(classifier, X, must_link, cannot_link):
    """
    Restricted Set Classification for the instances with pairwise constraints. 
    The class probability for each instance is defined as the mean of the probabilities reported by the classifier according to the must-link constraint.

    Parameters
    ----------
    classifier : ClassifierMixin with predict_proba method
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Array representing the data.
    must_link : dict of {int: list of int}
        Dictionary with the must links, where the key is the instance and the value is a list of instances that must have the same label.
    cannot_link : dict of {int: list of int}
        Dictionary with the cannot links, where the value is a list of instances that cannot have the same label.

    Returns
    ----------
    y : ndarray of shape (n_samples,)
        Array with predicted labels.

    Examples
    ----------
    ```python
    from sslearn.restricted import feature_fusion
    from sklearn.bayes import GaussianNB
    import pandas as pd

    dataset = pd.read_csv("dataset.csv")

    must_link = pd.read_csv("must_link.csv", index_col=0).to_dict(orient='index')
    # must_link = {0: [0, 2], 1: [1, 3]} -> 
    # instances 0 and 2 must have the same label, 
    # and instances 1 and 3 must have the same label

    cannot_link = pd.read_csv("cannot_link.csv", index_col=0).to_dict(orient='index')
    # cannot_link = {0: [0, 1], 1: [2, 3]} ->
    # instances 0 and 1 cannot have the same label, 
    # and instances 2 and 3 cannot have the same label

    X, y = dataset.iloc[:, :-1].values, dataset.iloc[:, -1].values
    X_label = X[y != y.dtype.type(-1)]
    y_label = y[y != y.dtype.type(-1)]
    X_unlabel = X[y == y.dtype.type(-1)]

    classifier = GaussianNB()
    classifier.fit(X_label, y_label)

    y_pred = probability_fusion(classifier, X_unlabel, must_link, cannot_link)
    ```

    References
    ----------
    L.I. Kuncheva, J.L. Garrido-Labrador, I. Ramos-Pérez, S.L. Hennessey, J.J. Rodríguez (2024).<br>
    Semi-supervised classification with pairwise constraints: A case study on animal identification from video.<br>
    <i>Information Fusion,</i><br> 
    104, 102188, [10.1016/j.inffus.2023.102188](https://doi.org/10.1016/j.inffus.2023.102188)
    """

    y_probs = classifier.predict_proba(X)
    classes = classifier.classes_
    y_probs_combined, _ = __combine_probabilities(y_probs, must_link, classes)
    return __restricted_set_classification(y_probs_combined, cannot_link, classes)


class WhoIsWhoClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):

    def __init__(self, base_estimator, method="hungarian", conflict_weighted=True):
        """
        Who is Who Classifier
        Kuncheva, L. I., Rodriguez, J. J., & Jackson, A. S. (2017).
        Restricted set classification: Who is there?. <i>Pattern Recognition</i>, 63, 158-170.

        Parameters
        ----------
        base_estimator : ClassifierMixin
            The base estimator to be used for training.
        method : str, optional
            The method to use to assing class, it can be `greedy` to first-look or `hungarian` to use the Hungarian algorithm, by default "hungarian"
        conflict_weighted : bool, default=True
            Whether to weighted the confusion rate by the number of instances with the same group.
        """        
        allowed_methods = ["greedy", "hungarian"]
        self.base_estimator = base_estimator
        self.method = method
        if method not in allowed_methods:
            raise ValueError(f"method {self.method} not supported, use one of {allowed_methods}")
        self.conflict_weighted = conflict_weighted


    def fit(self, X, y, instance_group=None, **kwards):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        instance_group : array-like of shape (n_samples)
            The group. Two instances with the same label are not allowed to be in the same group. If None, group restriction will not be used in training.
        Returns
        -------
        self : object
            Returns self.
        """
        self.base_estimator = self.base_estimator.fit(X, y, **kwards)
        self.classes_ = self.base_estimator.classes_
        if instance_group is not None:
            self.conflict_in_train = conflict_rate(self.base_estimator.predict(X), instance_group, self.conflict_weighted)
        else:
            self.conflict_in_train = None
        return self

    def conflict_rate(self, X, instance_group):
        """Calculate the conflict rate of the model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        instance_group : array-like of shape (n_samples)
            The group. Two instances with the same label are not allowed to be in the same group.
        Returns
        -------
        float
            The conflict rate.
        """
        y_pred = self.base_estimator.predict(X)
        return conflict_rate(y_pred, instance_group, self.conflict_weighted)

    def predict(self, X, instance_group):
        """Predict class for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        **kwards : array-like of shape (n_samples)
            The group. Two instances with the same label are not allowed to be in the same group.
        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        
        y_prob = self.predict_proba(X)
        
        y_predicted = combine_predictions(y_prob, instance_group, len(self.classes_), self.method)

        return self.classes_.take(y_predicted)


    def predict_proba(self, X):
        """Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        return self.base_estimator.predict_proba(X)


def conflict_rate(y_pred, restrictions, weighted=True):
    """
    Computes the conflict rate of a prediction, given a set of restrictions.
    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
    restrictions : array-like of shape (n_samples,)
        Restrictions for each sample. If two samples have the same restriction, they cannot have the same y.
    weighted : bool, default=True
        Whether to weighted the confusion rate by the number of instances with the same group.
    Returns
    -------
    conflict rate : float
        The conflict rate.
    """
    
    # Check that y_pred and restrictions have the same length
    if len(y_pred) != len(restrictions):
        raise ValueError("y_pred and restrictions must have the same length.")
    
    restricted_df = pd.DataFrame({'y_pred': y_pred, 'restrictions': restrictions})

    conflicted = restricted_df.groupby('restrictions').agg({'y_pred': lambda x: np.unique(x, return_counts=True)[1][np.unique(x, return_counts=True)[1]>1].sum()})
    if weighted:
        return conflicted.sum().y_pred / len(y_pred)
    else:
        rcount = restricted_df.groupby('restrictions').count()
        return (conflicted.y_pred / rcount.y_pred).sum()

def combine_predictions(y_probas, instance_group, class_number, method="hungarian"):
    """
    Combine the predictions of a group of instances to keep the restrictions.

    Parameters
    ----------
    y_probas : array-like of shape (n_samples, n_classes)
        The class probabilities of the input samples.
    instance_group : array-like of shape (n_samples)
        The group. Two instances with the same label are not allowed to be in the same group.
    class_number : int
        The number of classes.
    method : str, optional
        The method to use to assing class, it can be `greedy` to first-look or `hungarian` to use the Hungarian algorithm, by default "hungarian"

    Returns
    -------
    array-like of shape (n_samples,)
        The predicted labels.
    """
    y_predicted = []
    for group in np.unique(instance_group):
           
        mask = instance_group == group
        probas_matrix = y_probas[mask]
        

        preds = list(np.argmax(probas_matrix, axis=1))

        if len(preds) == len(set(preds)) or probas_matrix.shape[0] > class_number:
            y_predicted.extend(preds)
            if probas_matrix.shape[0] > class_number:
                warnings.warn("That the number of instances in the group is greater than the number of classes.", UserWarning)
            continue

        if method == "greedy":
            y = _greedy(probas_matrix)
        elif method == "hungarian":
            y = _hungarian(probas_matrix)
        
        y_predicted.extend(y)
    return y_predicted

def _greedy(probas_matrix):        

    probas = probas_matrix.reshape(probas_matrix.size,)
    order = probas.argsort()[::-1]

    y_pred_group = [None for i in range(probas_matrix.shape[0])]

    instance_to_predict = {i for i in range(probas_matrix.shape[0])}
    class_predicted = set()
    for item in order:
        class_ = item % probas_matrix.shape[0]
        instance = item // probas_matrix.shape[0]
        if instance in instance_to_predict and class_ not in class_predicted:
            y_pred_group[instance] = class_
            instance_to_predict.remove(instance)
            class_predicted.add(class_)
            
    return y_pred_group
        

def _hungarian(probas_matrix):
    
    costs = np.log(probas_matrix)
    costs[costs == -np.inf] = 0  # if proba is 0, then the cost is 0
    _, col_ind = linear_sum_assignment(costs, maximize=True)
    col_ind = list(col_ind)
        
    return col_ind

def __combine_probabilities(y_probs, objects_in_track, classes):
    """"
    Averages the classifier probabilities of the instances in the same track.
    
    :param y_probs: classifier probabilities for the instances 
    :param objects_in_track: dictionary with the tracks
    :param classes: classes used to train the classifier
    
    :return: a tuple with the modified y_probs ans the predicted classes
    """
    
    y_probs_combined = y_probs.copy()
    
    for objects in objects_in_track.values():
        if len(objects) <= 1:
            continue         
        means = y_probs_combined[objects, :].mean(axis=0)
        y_probs_combined[objects, :] = means  

    preds = classes.take(list(np.argmax(y_probs_combined, axis=1)))
    return y_probs_combined, preds

def __combine_features(X, objects_in_track):
    """
    Averages the features of the instances in the same track.
    
    :param X: feature values of the instances.
    :param objects_in_track: dictionary with the tracks
    
    :return: a modified X with averaged features
    """
    
    X_combined = X.copy()
    is_df = isinstance(X, pd.DataFrame)
    if is_df:
        X_combined = X.values
    for objects in objects_in_track.values():
        if len(objects) <= 1:
            continue
        means = X_combined[objects].mean(axis=0)
        X_combined[objects] = means 
    return X_combined

def __restricted_set_classification(y_probs, instances_by_frame, classes):
    """
    Restricted Set Classification for the instances in several frames
    
    :param y_probs: the probabilities given by the classifier for the instances
    :param instances_by_frame: which instances are in each frame
    :param classes: the classes seen by the classifier
    
    :return: the predicted labels
    """

    restricted_pred = []
    num_conflicts = 0
    for fr, group in instances_by_frame.items():
        if len(group) == 0:
            continue
        first, last = group[0], group[-1]
        group_probs = y_probs[first:last + 1]
        conflict, group_pred = __restricted_set_hungarian(group_probs, classes)
        restricted_pred.extend(group_pred)
        num_conflicts += conflict

    assert len(restricted_pred) == len(y_probs), "The number of predictions is different from the number of instances, check cannot link constraints, all instances must be in a cannot-link group."
        
    return restricted_pred

def __restricted_set_hungarian(probs, classes):
    """
    Restricted Set Classification for a set of objects that have to be of different classes
    
    :param probs: the probabilities given by the classifier
    :param classes: the classes seen by the classifier
    
    :return: a tuple with 1) the Hungarian method was used (0 or 1), and 2) the predicted classes
    """
    
    rows, cols = probs.shape
    preds = list(np.argmax(probs, axis=1))

    if rows > cols or len(preds) == len(set(preds)):
        # return 0 if rows > cols else 1, classes.take(preds)
        return 0, classes.take(preds)
    costs = np.log(probs)    
        
    try:
        row_ind, col_ind = linear_sum_assignment(costs, maximize=True)
        col_ind = list(col_ind)
    except: # some of the values was -Inf
        probs += np.nextafter(0, 1) # small double value
        costs = np.log(probs)        
        row_ind, col_ind = linear_sum_assignment(costs, maximize=True)
        col_ind = list(col_ind)
        
    return 1, classes.take(col_ind)
