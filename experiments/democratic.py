import numpy as np
import pandas as pd


import os
import sys
import warnings
from sklearn.base import clone as skclone

sys.path.insert(1, "..")

from sslearn.datasets import read_keel
import sslearn.wrapper as wrp
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

data_it = [
    "nursery",
    "iris",
    "banana",
    "autos"]

path = "/home/jlgarridol/Dropbox/GitHub/sslearn/data"
datasets = {}
for d in data_it:
    datasets[d] = list()
    for i in range(1, 11):
        data_path = os.path.join(path, f"{d}-ssl10")
        train = read_keel(os.path.join(data_path, f"{d}-ssl10-{i}tra.dat"), format="numpy")
        trans = read_keel(os.path.join(data_path, f"{d}-ssl10-{i}trs.dat"), format="numpy")
        test = read_keel(os.path.join(data_path, f"{d}-ssl10-{i}trs.dat"), format="numpy")
        datasets[d].append((train, trans, test))

seed = 100
classifier_seed = 0

classifiers = {
    "Democratic": wrp.DemocraticCoLearning(base_estimator=[
        DecisionTreeClassifier(random_state=classifier_seed, min_samples_leaf=2),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=3)], confidence_method="bernoulli")
}

def experiment():
    print("Start experiments")
    warnings.filterwarnings("ignore")

    acc_trans, acc_ind = dict(), dict()
    for c in classifiers:
        acc_trans[c] = dict()
        acc_ind[c] = dict()
        for d in datasets:
            acc_trans[c][f"{d}-ssl10"] = list()
            acc_ind[c][f"{d}-ssl10"] = list()

    for d in data_it:
        print("Processing with", d)

        for c in classifiers:
            print("\tTraining", c)
            for i in range(10):
                (X_train, y_train), (X_trans, y_trans), (X_test, y_test) = datasets[d][i]
                learner = skclone(classifiers[c])

                learner.fit(X_train, y_train)

                score_trans = learner.score(X_trans, y_trans)
                score_ind = learner.score(X_test, y_test)

                acc_trans[c][f"{d}-ssl10"].append(score_trans)
                acc_ind[c][f"{d}-ssl10"].append(score_ind)

            del learner

    print("End experiment")


# for lr in label_rates:
#     experiment(lr)
experiment()