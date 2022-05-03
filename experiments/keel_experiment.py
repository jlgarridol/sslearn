
import gc
import joblib as jl
import pickle
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(1, "..")

import warnings
from termcolor import colored
from sklearn.utils import check_random_state as crs
from sklearn.base import clone as skclone
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import sslearn.wrapper as wrp

import pathlib
import curses
from sslearn.datasets import read_keel

no_ssl = False

save_path = pathlib.Path(__file__).parent.resolve()
path = "/home/jlgarridol/Dropbox/GitHub/sslearn/data"

data_it = [
    "abalone",
    "appendicitis",
    "australian",
    "autos",
    "balance",
    "banana",
    "bands",
    "breast",
    "bupa",
    "car",
    "chess",
    "cleveland",
    "coil2000",
    "contraceptive",
    "crx",
    "dermatology",
    "ecoli",
    "flare-solar",
    "german",
    "glass",
    "haberman",
    "hayes-roth",
    "heart",
    "hepatitis",
    "housevotes",
    "ionosphere",
    "iris",
    "kr-vs-k",
    "led7digit",
    "letter",
    "lymphography",
    "magic",
    "mammographic",
    "marketing",
    "monks",
    "movement_libras",
    "mushroom",
    "newthyroid",
    "nursery",
    "optdigits",
    "pagebloks",
    "penbased",
    "phoneme",
    "pima",
    "post-operative",
    "ring",
    "saheart",
    "satimage",
    "segment",
    "shuttle",
    "sonar",
    "spambase",
    "spectfheart",
    "splice",
    "tae",
    "texture",
    "thyroid",
    "tic-tac-toe",
    "titanic",
    "twonorm",
    "vehicle",
    "vowel",
    "wdbc",
    "wine",
    "wisconsin",
    "yeast",
    "zoo",
]

modes = [10, 20, 30, 40]

data_sizes = {}
for m in modes:
    datasets = {}
    for d in data_it:
        datasets[d] = list()
        for i in range(1, 11):
            data_path = os.path.join(path, f"{d}-ssl{m}")
            train = read_keel(os.path.join(data_path, f"{d}-ssl{m}-{i}tra.dat"), format="numpy")
            trans = read_keel(os.path.join(data_path, f"{d}-ssl{m}-{i}trs.dat"), format="numpy")
            test = read_keel(os.path.join(data_path, f"{d}-ssl{m}-{i}trs.dat"), format="numpy")
            datasets[d].append((train, trans, test))
    data_sizes[m] = datasets

print("Datasets loaded")

seed = 100
classifier_seed = 0

classifiers = {
    # "CoBC": wrp.CoTrainingByCommittee(ensemble_estimator=BaggingClassifier(
    #     n_estimators=3, base_estimator=DecisionTreeClassifier(
    #         random_state=classifier_seed, min_samples_leaf=2),
    #     random_state=classifier_seed),
    #     max_iterations=30, poolsize=100, random_state=classifier_seed),
    # "TriTraining": wrp.TriTraining(base_estimator=DecisionTreeClassifier(
    #     random_state=classifier_seed, min_samples_leaf=2),
    #     random_state=classifier_seed),
    "Democratic": wrp.DemocraticCoLearning(base_estimator=[
        DecisionTreeClassifier(random_state=classifier_seed, min_samples_leaf=2),
        GaussianNB(),
        KNeighborsClassifier(n_neighbors=3)], confidence_method="bernoulli")
    # "NNSSL": KNeighborsClassifier(n_neighbors=1, n_jobs=-1),
    # "NBSSL": GaussianNB(),
    # "C45SSL": DecisionTreeClassifier(random_state=classifier_seed, min_samples_leaf=2),
    # "SMOSSL": SVC(random_state=classifier_seed, C=100, tol=0.001, kernel="rbf", gamma=0.01)
}


def experiment():
    print("Start experiments")
    warnings.filterwarnings("ignore")

    acc_trans, acc_ind = dict(), dict()
    for c in classifiers:
        acc_trans[c] = dict()
        acc_ind[c] = dict()
        for m in modes:
            for d in data_sizes[m]:
                acc_trans[c][f"{d}-ssl{m}"] = list()
                acc_ind[c][f"{d}-ssl{m}"] = list()

    for m in modes:
        print("Processing with size", m)
        for d in data_it:
            print("Processing with", d)

            for c in classifiers:
                print("\tTraining", c)
                for i in range(10):
                    (X_train, y_train), (X_trans, y_trans), (X_test, y_test) = data_sizes[m][d][i]
                    learner = skclone(classifiers[c])

                    if no_ssl:
                        X, y = X_train[y_train != y_train.dtype(-1)], y_train[y_train != y_train.dtype(-1)]
                    else:
                        X, y = X_train, y_train
                    learner.fit(X, y)

                    score_trans = learner.score(X_trans, y_trans)
                    score_ind = learner.score(X_test, y_test)

                    acc_trans[c][f"{d}-ssl{m}"].append(score_trans)
                    acc_ind[c][f"{d}-ssl{m}"].append(score_ind)

                del learner
                gc.collect()

    with open(os.path.join(save_path, "democratic_trans_sslearn.pkl"), "wb") as f:
        pickle.dump(acc_trans, f)
    with open(os.path.join(save_path, "democratic_ind_sslearn.pkl"), "wb") as f:
        pickle.dump(acc_ind, f)

    print("End experiment")


# for lr in label_rates:
#     experiment(lr)
experiment()
