
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

import sslearn.wrapper as wrp

import pathlib
import curses
from sslearn.datasets import read_keel

save_path = pathlib.Path(__file__).parent.resolve()
path = "/home/jlgarridol/Dropbox/GitHub/sslearn/data"

data_it = [
    "vowel",
    "flare-solar",
    "banana",
    "autos",
    "titanic",
    "contraceptive",
    "monks",
    "thyroid",
    "appendicitis",
    "saheart",
    "kr-vs-k",
    "movement_libras",
    "letter",
    "hepatitis",
    "segment",
    "satimage",
    "nursery",
    "led7digit",
    "chess",
    "pima",
    "abalone",
    "car",
    "shuttle",
    "mammographic",
    "lymphography",
    "optdigits",
    "penbased",
    "glass",
    "dermatology",
    "sonar",
    "bands",
    "vehicle",
    "iris",
    "spectfheart",
    "german",
    "pagebloks",
    "housevotes",
    "haberman",
    "australian",
    "ecoli",
    "texture",
    "yeast",
    "crx",
    "balance",
    "marketing",
    "bupa",
    "mushroom",
    "tae",
    "breast",
    "tic-tac-toe",
    "coil2000",
    "spambase",
    "wine",
    "ionosphere",
    "hayes-roth",
    "phoneme",
    "wdbc",
    "splice",
    "newthyroid",
    "twonorm",
    "zoo",
    "ring",
    "cleveland",
    "post-operative",
    "wisconsin",
    "heart",
    "magic",
]

datasets = {}
for d in data_it:
    datasets[d] = list()
    for i in range(1, 11):
        data_path = os.path.join(path, f"{d}-ssl10")
        train = read_keel(os.path.join(data_path, f"{d}-ssl10-{i}tra.dat"), format="numpy")
        trans = read_keel(os.path.join(data_path, f"{d}-ssl10-{i}trs.dat"), format="numpy")
        test = read_keel(os.path.join(data_path, f"{d}-ssl10-{i}trs.dat"), format="numpy")
        datasets[d].append((train, trans, test))

print("Datasets loaded")

seed = 100
classifier_seed = 0

classifiers = {
    "CoBC": wrp.CoTrainingByCommittee(ensemble_estimator=BaggingClassifier(
        n_estimators=3, base_estimator=DecisionTreeClassifier(
            random_state=classifier_seed, min_samples_leaf=2),
        random_state=classifier_seed),
        max_iterations=30, poolsize=100, random_state=classifier_seed),
    "TriTraining": wrp.TriTraining(base_estimator=DecisionTreeClassifier(
        random_state=classifier_seed, min_samples_leaf=2),
        random_state=classifier_seed),
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
            gc.collect()

    with open(os.path.join(save_path, "acc_trans_sslearn" + ".pkl"), "wb") as f:
        pickle.dump(acc_trans, f)
    with open(os.path.join(save_path, "acc_ind_sslearn" + ".pkl"), "wb") as f:
        pickle.dump(acc_ind, f)

    print("End experiment")


# for lr in label_rates:
#     experiment(lr)
experiment()
