import sys
import os

import joblib as jl
import numpy as np
import pytest
import scipy.stats as st
from pytest import approx
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.semi_supervised import SelfTrainingClassifier
from statsmodels.stats.proportion import proportion_confint

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sslearn.base import FakedProbaClassifier, OneVsRestSSLClassifier
from sslearn.restricted import (WhoIsWhoClassifier, combine_predictions,
                                conflict_rate, probability_fusion, feature_fusion)
from sslearn.utils import (calc_number_per_class, calculate_prior_probability,
                           check_n_jobs, choice_with_proportion,
                           confidence_interval, is_int, safe_division)


class TestUtils():
    def test_is_int(self):
        assert is_int(1)
        assert not is_int(1.0)
        assert not is_int(1.1)
        assert not is_int('1')

    def test_safe_division(self):
        assert safe_division(1, 2, sys.float_info.epsilon) == 0.5
        assert safe_division(1, 0, sys.float_info.epsilon) == 1/sys.float_info.epsilon

    def test_calc_number_per_class(self):
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert calc_number_per_class(y) == {0: 1, 1: 1, 2: 1}

    def test_choice_with_proportion(self):
        hyp = GaussianNB()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])
        hyp.fit(X, y)
        proportion = {0: 0.5, 1: 0.5}

        assert choice_with_proportion(np.max(hyp.predict_proba(X), axis=1), hyp.predict(X), proportion).tolist() == [4, 2, 0, 1]

    def test_confidence_interval(self):
        hyp = DummyClassifier()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])
        hyp.fit(X, y)
        data = hyp.predict(X)

        li, hi = proportion_confint(
            np.count_nonzero(data == y), X.shape[0], alpha=0.05, method="wilson")

        assert confidence_interval(X, hyp, y) == (approx(li), approx(hi))

    def test_check_n_jobs(self):
        assert check_n_jobs(1) == 1
        assert check_n_jobs(-1) == jl.cpu_count()
        assert check_n_jobs(2) == 2
        assert check_n_jobs(None) == 1

    def test_calculate_prior_probability(self):
        y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        assert calculate_prior_probability(y) == {0: 0.3333333333333333,
                                                  1: 0.3333333333333333,
                                                  2: 0.3333333333333333}
        y = [0, 0, 1, 1, 1, 2, 3, 3, 3]
        assert calculate_prior_probability(y) == {0: 0.2222222222222222,
                                                  1: 0.3333333333333333, 
                                                  2: 0.1111111111111111,
                                                  3: 0.3333333333333333}
                                                    
class TestBase():

    def test_FakedProbaClassifier(self):
        hyp = FakedProbaClassifier(LinearSVC())
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 1, 1])
        hyp.fit(X, y)
        assert hyp.predict(X).tolist() == [0, 0, 0, 1, 1]
        assert hyp.predict_proba(X).tolist() == [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
        assert hyp.score(X, y) == 1.0

    def test_OneVsRestSSLClassifier(self):
        hyp = OneVsRestSSLClassifier(SelfTrainingClassifier(GaussianNB()))
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 2, -1, -1])
        hyp.fit(X, y)
        assert hyp.predict(X).tolist() == [0, 1, 2, 2, 2]

class TestRestricted():
        
    def test_conflict_rate(self):
        pred = np.array([1, 2, 3])
        rest = np.array([1, 2])

        with pytest.raises(ValueError):
            conflict_rate(pred, rest)

        pred = np.array([1, 1, 1, 2, 2, 2, 1, 2])
        rest = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        assert conflict_rate(pred, rest) == 0.5
        assert conflict_rate(pred, rest, True) == 0.5

    def test_combine_predictions(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 2, 1, 2])
        group = np.array([0, 0, 1, 1, 1])
        hyp = GaussianNB()
        hyp.fit(X, y)
        pred = hyp.predict_proba(X)
        greedy = combine_predictions(pred, group, len(np.unique(y)), 'greedy')
        hungarian = combine_predictions(pred, group, len(np.unique(y)), 'hungarian')

        assert greedy == [0, 1, 1, 0, 2]
        assert hungarian == [0, 1, 1, 0, 2]

    def test_WhoIsWhoClassifier(self):
        hyp = WhoIsWhoClassifier(DummyClassifier())
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 0, 1, 2])
        group = np.array([0, 0, 1, 1, 1])
        hyp.fit(X, y, group)

        assert hyp.conflict_in_train == 1
        assert hyp.predict(X, group).tolist() == [0, 1, 0, 2, 1]

    def test_probability_fusion(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 0, 1, 2])
        cannot_link = {0: [0, 1], 1: [2, 3, 4]}
        must_link = {1: [1, 3], 0: [0, 2], 4: [4]}

        h = GaussianNB()
        h.fit(X, y)

        probability_fusion(h, X, must_link=must_link, cannot_link=cannot_link)

    def test_feature_fusion(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 0, 1, 2])
        cannot_link = {0: [0, 1], 1: [2, 3, 4]}
        must_link = {1: [1, 3], 0: [0, 2], 4: [4]}

        h = GaussianNB()
        h.fit(X, y)

        result = feature_fusion(h, X, must_link=must_link, cannot_link=cannot_link)



