import os
import sys
import pandas as pd
import numpy as np

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

from sslearn.datasets import read_csv
from sslearn.model_selection import artificial_ssl_dataset
from sslearn.wrapper import (
    CoTraining, CoForest, CoTrainingByCommittee, DemocraticCoLearning, Rasco, RelRasco,
    SelfTraining, Setred, TriTraining, WiWTriTraining, DeTriTraining
)

X, y = read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "example_files", "abalone.csv"), format="pandas")
X2, y2 = read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "example_files", "abalone.csv"), format="numpy")

X_l, y_l = load_breast_cancer(return_X_y=True)


multiples_estimator = [
            DecisionTreeClassifier(max_depth=1),
            DecisionTreeClassifier(max_depth=2),
            KNeighborsClassifier(n_neighbors=5),
            GaussianNB(),
            KNeighborsClassifier(n_neighbors=3),
        ]

def check_random(estimator, estimator_key="base_estimator", **kwargs):
    for i in range(5):
        clf = eval(f"estimator({estimator_key}=KNeighborsClassifier(), random_state=i, **kwargs)")
        clf.fit(X, y)
        y1 = clf.predict(X)
        clf = eval(f"estimator({estimator_key}=KNeighborsClassifier(), random_state=i, **kwargs)")
        clf.fit(X, y)
        y2 = clf.predict(X)
        assert np.all(y1 == y2)

def check_multiple(estimator, **kwargs):
    clf = estimator(base_estimator=multiples_estimator, n_estimators=5, **kwargs)
    clf.fit(X, y)
    clf.predict(X)
    clf.predict_proba(X)

def check_all_label(estimator, **kwargs):
    clf = estimator(**kwargs)
    clf.fit(X_l, y_l)
    clf.predict(X_l)
    clf.predict_proba(X_l)

def check_pandas(estimator, **kwargs):
    clf = estimator(**kwargs)
    clf.fit(X, y)
    clf.predict(X)
    clf.predict_proba(X)

def check_numpy(estimator, **kwargs):
    clf = estimator(**kwargs)
    clf.fit(X2, y2)
    clf.predict(X2)
    clf.predict_proba(X2)

def check_basic(estimator, **kwargs):
    check_pandas(estimator, **kwargs)
    check_numpy(estimator, **kwargs)


class TestCoForest:

    def test_basic(self):
        check_basic(CoForest)

    def test_multiple(self):
        check_multiple(CoForest)

    def test_random_state(self):
        check_random(CoForest)

    def test_all_label(self):
        check_all_label(CoForest)
    

class TestCoTrainingByCommittee:

    def test_basic(self):
        check_basic(CoTrainingByCommittee)

    def test_random_state(self):
        check_random(CoTrainingByCommittee, estimator_key="ensemble_estimator")

    def test_all_label(self):
        check_all_label(CoTrainingByCommittee)


class TestCoTraining:

    def test_basic(self):
        check_basic(CoTraining, force_second_view=False)

    def test_multiview(self):
        clf = CoTraining()
        clf.fit(X.iloc[:,:len(X.columns)//2], y, X2=X.iloc[:, len(X.columns)//2:])
        clf.predict(X.iloc[:, :len(X.columns)//2], X2=X.iloc[:, len(X.columns)//2:])
        clf.predict_proba(X.iloc[:, :len(X.columns)//2], X2=X.iloc[:, len(X.columns)//2:])

        clf = CoTraining()
        clf.fit(X, y, features=[[i for i in range(len(X.columns)//2)], [i for i in range(len(X.columns)//2, len(X.columns))]])
        clf.predict(X)
        clf.predict_proba(X)

    def test_multiple(self):
        clf = CoTraining(force_second_view=False, 
        base_estimator=DecisionTreeClassifier(max_depth=1),
        second_base_estimator=DecisionTreeClassifier(max_depth=2))
        clf.fit(X, y)
        clf.predict(X)
        clf.predict_proba(X)

    def test_random_state(self):
        check_random(CoTraining, force_second_view=False)


    def test_all_label(self):
        check_all_label(CoTraining, force_second_view=False)


class TestDemocraticCoLearning:
    
    def test_basic(self):
        check_basic(DemocraticCoLearning)

    def test_multiple(self):
        check_multiple(DemocraticCoLearning)
    
    def test_random_state(self):
        check_random(DemocraticCoLearning, n_estimators=3)

    def test_all_label(self):
        check_all_label(DemocraticCoLearning)


class TestRasco:

    def test_basic(self):
        check_basic(Rasco)

    def test_multiple(self):
        check_multiple(Rasco)

    def test_random_state(self):
        check_random(Rasco, n_estimators=10)

    def test_all_label(self):
        check_all_label(Rasco)


class TestRelRasco:

    def test_basic(self):
        check_basic(RelRasco)

    def test_multiple(self):
        check_multiple(RelRasco)

    def test_random_state(self):
        check_random(RelRasco, n_estimators=10)

    def test_all_label(self):
        check_all_label(RelRasco)


class TestSelfTraining:

    def test_basic(self):
        check_basic(SelfTraining, base_estimator=KNeighborsClassifier())

    def test_all_label(self):
        check_all_label(SelfTraining, base_estimator=KNeighborsClassifier())



class TestSetred:

    def test_basic(self):
        check_basic(Setred)

    def test_random_state(self):
        check_random(Setred)

    def test_all_label(self):
        check_all_label(Setred)


class TestTriTraining:
    
    def test_basic(self):
        check_basic(TriTraining)
    
    def test_random_state(self):
        check_random(TriTraining)

    def test_multiple(self):
        clf = TriTraining(base_estimator=[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)])
        clf.fit(X, y)
        clf.predict(X)
        clf.predict_proba(X)

    def test_no_more_three(self):
        with pytest.raises(AttributeError):
            _ =  TriTraining(base_estimator=multiples_estimator)

    def test_all_label(self):
        check_all_label(TriTraining)
    

class TestDeTriTraining:
    
    def test_basic(self):
        check_basic(DeTriTraining, max_iterations=1)
    
    def test_random_state(self):
        check_random(DeTriTraining, max_iterations=1)

    def test_all_label(self):
        check_all_label(DeTriTraining, max_iterations=1)

# Create a fake groups
groups = list()
for order, i in enumerate(range(0, len(X), 5)):
    groups.extend([order for _ in range(5)])
groups = np.array(groups)
groups = groups[:X.shape[0]]

class TestWiWTriTraining:
    
    def test_basic(self):
        clf = WiWTriTraining(base_estimator=DecisionTreeClassifier())
        clf.fit(X, y, instance_group=groups)
        clf.predict(X, instance_group=groups)
        clf.predict_proba(X)

        clf = WiWTriTraining(DecisionTreeClassifier())
        clf.fit(X2, y2, instance_group=groups)
        clf.predict(X2, instance_group=groups)
        clf.predict_proba(X2)

    def test_multiple(self):
        clf = WiWTriTraining(base_estimator=[DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2), DecisionTreeClassifier(max_depth=3)])
        clf.fit(X, y, instance_group=groups)
        clf.predict(X, instance_group=groups)
        clf.predict_proba(X)
    
    def test_random_state(self):
        for i in range(10):
            clf = WiWTriTraining(base_estimator=KNeighborsClassifier(), random_state=i)
            clf.fit(X, y, instance_group=groups)
            y1 = clf.predict(X, instance_group=groups)

            clf = WiWTriTraining(base_estimator=KNeighborsClassifier(), random_state=i)
            clf.fit(X, y, instance_group=groups)
            y2 = clf.predict(X, instance_group=groups)

            assert np.all(y1 == y2)

    def test_all_label(self):
        clf = WiWTriTraining(base_estimator=KNeighborsClassifier())
        clf.fit(X, y, instance_group=groups)
        clf.predict(X, instance_group=groups)
        clf.predict_proba(X)

