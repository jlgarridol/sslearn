
import gc
import os
import pickle
import sys

import joblib as jl
import numpy as np
import pandas as pd

sys.path.insert(1, "..")

import curses
import pathlib
import warnings

import sslearn.wrapper as wrp
from sklearn.base import clone as skclone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state as crs
from sktime.contrib.vector_classifiers._rotation_forest import \
    RotationForest as RotationForestSktime
from sslearn.model_selection import artificial_ssl_dataset
from sslearn.supervised.rotation import RotationForestClassifier
from termcolor import colored
from sslearn.datasets import read_csv

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

seed = 100
n_splits = 10
classifier_seed = 0
repetitions = 5
label_rates = [x / 10 for x in range(1, 5)]
colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "white", "gray"]

classifiers = {
    # "RandomForest": RandomForestClassifier(random_state=classifier_seed, n_jobs=-1),
    "DemocraticCoLearning": wrp.DemocraticCoLearning(),
    # "SelfTraining": wrp.SelfTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1)),
    # "TriTraining-Old": wrp.TriTraining(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed, mode="old"),
    # "TriTraining-New": wrp.TriTraining(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed, mode="new"),
    # "Rasco": wrp.Rasco(base_estimator=RandomForestClassifier(n_estimators=10, random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed, n_jobs=-1),
    # "RelRasco": wrp.RelRasco(base_estimator=RandomForestClassifier(n_estimators=10, random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed, n_jobs=-1),
    # "CoForest": wrp.CoForest(random_state=classifier_seed),
    # "DeTriTraining": wrp.DeTriTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    # "CoTrainingByComitte": wrp.CoTrainingByCommittee(ensemble_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    # "CoTraining": OneVsRestClassifier(wrp.CoTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1)))
}


def experiment(lr):
    # print("\nLabel rate: {}".format(int(lr*100)))
    warnings.filterwarnings("ignore")
    color_index = label_rates.index(lr)

    acc_trans, acc_ind = dict(), dict()
    for c in classifiers:
        acc_trans[c] = dict()
        acc_ind[c] = dict()
        for d in datasets:
            acc_trans[c][d] = list()
            acc_ind[c][d] = list()

    steps = len(classifiers) * len(datasets) * repetitions
    step = 0
    for i, d in enumerate(data_it):

        X, y = datasets[d]

        for c in classifiers:            
            for r in range(repetitions):
                step += 1

                skf = StratifiedKFold(
                    n_splits=n_splits, random_state=seed * r, shuffle=True
                )
                # cols = curses.tigetnum('cols')
                text = "Learning Rate {}, Classifier {}, Dataset {}, Repetition {}. Steps {}/{}".format(
                    lr, c, d, r + 1, step, steps
                )
                # points = cols-len(text)-len(vueltas)
                final_text = f"{text}"
                print(colored(final_text, colors[color_index]))
                for train, test in skf.split(X, y):

                    X_train, y_train = X[train], y[train]
                    X_test, y_test = X[test], y[test]

                    X_, y_, X_unlabel, y_true = artificial_ssl_dataset(
                        X_train,
                        y_train,
                        label_rate=lr,
                        random_state=seed * r,
                    )
                    learner = skclone(classifiers[c])
                    learner.random_state = classifier_seed + r
                    learner.fit(X_, y_)

                    score_trans = learner.score(X_unlabel, y_true)
                    score_ind = learner.score(X_test, y_test)

                    acc_trans[c][d].append(score_trans)
                    acc_ind[c][d].append(score_ind)

                    del X_
                    del y_
                    del X_unlabel
                    del y_true
                    del X_train
                    del y_train
                    del X_test
                    del y_test

            del learner
            gc.collect()

    print("Finalizado", lr * 100, file=sys.stderr)
    with open(os.path.join(save_path, "democo_acc_trans_" + str(int(lr * 100)) + ".pkl"), "wb") as f:
        pickle.dump(acc_trans, f)
    with open(os.path.join(save_path, "democo_acc_ind_" + str(int(lr * 100)) + ".pkl"), "wb") as f:
        pickle.dump(acc_ind, f)


# for lr in label_rates:
#     experiment(lr)
jl.Parallel(n_jobs=4)(jl.delayed(experiment)(lr) for lr in label_rates)
