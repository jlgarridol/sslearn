from math import isnan
from numpy.random import random_sample
from sklearn.base import ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.base import clone as skclone
from sklearn.utils import check_random_state, resample
import numpy as np
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import kneighbors_graph
from sslearn.utils import calculate_prior_probability
from scipy.stats import norm, bernoulli
import math


SelfTraining = SelfTrainingClassifier

class Setred(ClassifierMixin):

    def __init__(self, base_estimator=DecisionTreeClassifier(), max_iterations=40,
                       distance="euclidean", pool_size=0.25, rejection_threshold=0.1,
                       random_state=None, n_jobs=None):
        """[summary]

        Parameters
        ----------
        base_estimator : [type], optional
            [description], by default DecisionTreeClassifier()
        max_iterations : int, optional
            [description], by default 40
        distance : str, optional
            [description], by default "euclidean"
        pool_size : float, optional
            [description], by default 0.25
        rejection_threshold : float, optional
            [description], by default 0.1
        random_state : [type], optional
            [description], by default None
        n_jobs : [type], optional
            [description], by default None
        """
        self.base_estimator = skclone(base_estimator)
        self.max_iterations = max_iterations
        self.pool_size = pool_size
        self.distance = distance
        self.rejection_threshold = rejection_threshold
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs

    def __create_neighborhood(self, X):
        return kneighbors_graph(X, 1, metric=self.distance, n_jobs=self.n_jobs).toarray(), \
               kneighbors_graph(X, 1, metric=self.distance, n_jobs=self.n_jobs, mode='distance').toarray()


    def __calculate_J(self, X, p_y_y_, G, weights, instance):
        J=0
        for i in range(len(X)):
            if G[instance, i] == 1:
                J += weights[instance, i]*bernoulli.rvs(p_y_y_, random_state=self.random_state)
        return J

    def __calculate_H0(self, p_y_y_, G, weights, instance):

            U_ = resample(X_unlabel, replace=False,
        sum_w = weights[instance, G[instance,:]==1].sum()
        sum_sq_w = sum(map(lambda x: x**2,weights[instance, G[instance,:]==1]))

        mu_h0 = (1-p_y_y_)*sum_w
        sigma_h0 = p_y_y_*(1-p_y_y_)*sum_sq_w

        return mu_h0, sigma_h0        

    def fit(self, X, y, **kwars):
        X_label = X[y != y.dtype.type(-1)]
        y_label = y[y != y.dtype.type(-1)]
        X_unlabel = X[y == y.dtype.type(-1)]

        pool = int(len(X_label)*self.pool_size)

        self.base_estimator.fit(X_label, y_label, **kwars)
        for _ in range(self.max_iterations):
            U_ = resample(X_unlabel, replace=False,
                          n_samples=len(X_label),
                          random_state=self.random_state)

            # Se supone que solo necesito las kj más relevantes
            # Es número aún no se cual es
            # De momento cojo el 25% de L:
            raw_predictions = \
                self.base_estimator.predict_proba(U_)
            predictions = np.max(raw_predictions, axis=1)
            class_predicted = np.argmax(raw_predictions, axis=1)
            indexes = predictions.argsort()[-pool:]

            L_ = U_[indexes]
            y_ = np.array(list(map(lambda x: self.base_estimator.classes_[x],
                                        class_predicted[indexes])))

            pre_L = np.concatenate((X_label, L_), axis=0)

            G, weights = self.__create_neighborhood(pre_L)
            y_probabilities = calculate_prior_probability(y_label)
            to_add = list()
            for i, _ in enumerate(L_):
                i_plus = i+len(X_label)
                Ji = self.__calculate_J(pre_L, y_probabilities[y_[i]] , G, weights, i_plus)
                mu_, sigma_ = self.__calculate_H0(y_probabilities[y_[i]], G, weights, i)

                oi = norm(mu_, sigma_).pdf(Ji)

                if oi <= self.rejection_threshold:  # Es menor, por tanto está a la derecha (creo)
                    to_add.append(i)
                
            L_filtered = L_[to_add,:]
            y_filtered = y_[to_add]

            X_label = np.concatenate((X_label, L_filtered), axis=0)
            y_label = np.concatenate((y_label, y_filtered), axis=0)

            
        return self
    
    def predict(self, X, **kwards):
        return self.base_estimator.predict(X, **kwards)

    def predict_proba(self, X, **kwards):
        return self.base_estimator.predict_proba(X, **kwards)



