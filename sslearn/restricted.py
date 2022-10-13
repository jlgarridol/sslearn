import numpy as np
from sklearn.base import ClassifierMixin, MetaEstimatorMixin, BaseEstimator
from scipy.optimize import linear_sum_assignment
import warnings
import pandas as pd

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