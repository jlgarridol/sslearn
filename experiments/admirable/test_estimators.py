
import gc
import os
import pickle
import sys

import joblib as jl
import numpy as np
import pandas as pd

sys.path.insert(1, "../..")
sys.path.insert(1, "../../../admirable-methods/")

import curses
import pathlib
import warnings

from ubulearn.features import RFWClassifier
from ubulearn.neighbors import DisturbingNeighborsClassifier
import sslearn.wrapper as wrp
from sslearn.wrapper import SelfTraining, CoTrainingByCommittee
from sklearn.base import clone as skclone
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sslearn.model_selection import artificial_ssl_dataset
from termcolor import colored

save_path = pathlib.Path(__file__).parent.resolve()
path = "/home/jlgarridol/Dropbox/UBU/Investigación/SSRotation/csv"

datasets = {}
for file in os.listdir(path):
    dataset = pd.read_csv(os.path.join(path, file), header=None)

    columns = []
    for i, tp in enumerate(dataset.dtypes):
        if not np.issubdtype(tp, np.number) and i != dataset.shape[1] - 1:
            columns.append(i)
            dataset[i] = dataset[i].astype("|S")

    y = dataset.iloc[:, -1]
    if np.issubdtype(y, np.number):
        y = y + 2
    X = dataset.iloc[:, :-1]
    if len(columns) > 0:
        elements = [X[X.columns.difference(columns)]]

        for col in columns:
            elements.append(pd.get_dummies(X[col]))

        concatenated_data = pd.concat(elements, axis=1)
        X = concatenated_data
    datasets[file.split(".")[0]] = (X.to_numpy(), y.to_numpy())

data_it = [
    "abalone",
    # "appendicitis",
    # "australian",
    # "autos",
    # "balance",
    # "banana",
    # "bands",
    # "breast",
    # "bupa",
    # "car",
    # "chess",
    # "cleveland",
    # "coil2000",
    # "contraceptive",
    # "crx",
    # "dermatology",
    # "ecoli",
    # "flare-solar",
    # "german",
    # "glass",
    # "haberman",
    # "hayes-roth",
    # "heart",
    # "hepatitis",
    # "housevotes",
    # "ionosphere",
    # "iris",
    # "kr-vs-k",
    # "led7digit",
    "letter",
    # "lymphography",
    # "magic",
    # "mammographic",
    "marketing",
    # "monks",
    "movement_libras",
    # "mushroom",
    # "newthyroid",
    # "nursery",
    # "optdigits",
    # "pagebloks",
    # "penbased",
    # "phoneme",
    # "pima",
    # "post-operative",
    # "ring",
    # "saheart",
    # "satimage",
    # "segment",
    # "shuttle",
    # "sonar",
    # "spambase",
    # "spectfheart",
    # "splice",
    # "tae",
    # "texture",
    "thyroid",
    # "tic-tac-toe",
    # "titanic",
    # "twonorm",
    "vehicle",
    "vowel",
    # "wdbc",
    # "wine",
    # "wisconsin",
    "yeast",
    # "zoo",
]

seed = 100
n_splits = 10
classifier_seed = 0
repetitions = 5
label_rates = [x / 10 for x in range(1, 5)]
colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "white", "gray"]

classifiers = {
    "RFW": (RFWClassifier(n_estimators=100, n_jobs=None), "supervised"),
    "DisturbingNeighbors-Bag": (BaggingClassifier(DisturbingNeighborsClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed), n_estimators=100, n_jobs=None), "supervised"),
    "DisturbingNeighbors-Ada": (AdaBoostClassifier(DisturbingNeighborsClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed), n_estimators=100), "supervised"),
    "SelfRFW": (SelfTraining(RFWClassifier(n_estimators=100, n_jobs=None, random_state=classifier_seed)), "semi-supervised"),
    "SelfDisturbingNeighbors-Bag": (SelfTraining(BaggingClassifier(DisturbingNeighborsClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed),  random_state=classifier_seed), n_estimators=100, n_jobs=None)), "semi-supervised"),
    "SelfDisturbingNeighbors-Ada": (SelfTraining(AdaBoostClassifier(DisturbingNeighborsClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed),  random_state=classifier_seed), n_estimators=100)), "semi-supervised"),
    "CoTrainingByCommittee-RFW": (CoTrainingByCommittee(RFWClassifier(n_estimators=100, n_jobs=None, random_state=None)), "semi-supervised"),
    "CoTrainingByCommittee-DisturbingNeighbors-Bag": (CoTrainingByCommittee(BaggingClassifier(DisturbingNeighborsClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed), n_estimators=100, n_jobs=None)), "semi-supervised"),
    "CoTrainingByCommittee-DisturbingNeighbors-Ada": (CoTrainingByCommittee(AdaBoostClassifier(DisturbingNeighborsClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed), n_estimators=100)), "semi-supervised")
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

    steps = len(classifiers) * len(data_it) * repetitions
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
                    learner = skclone(classifiers[c][0])
                    learner.random_state = classifier_seed + r

                    if classifiers[c][1] == "semi-supervised":
                        learner.fit(X_, y_)
                    else:
                        learner.fit(X_[y_!=y.dtype.type(-1), :], y_[y_!=y.dtype.type(-1)])

                    score_trans = learner.score(X_unlabel, y_true)
                    score_ind = learner.score(X_test, y_test)

                    acc_trans[c][d].append(score_trans)
                    acc_ind[c][d].append(score_ind)

                    del X_, y_, X_unlabel, y_true, X_train, y_train, X_test, y_test

            del learner
            gc.collect()

    print("Finalizado", lr * 100, file=sys.stderr)
    with open(os.path.join(save_path, "admirable_acc_trans_" + str(int(lr * 100)) + ".pkl"), "wb") as f:
        pickle.dump(acc_trans, f)
    with open(os.path.join(save_path, "admirable_acc_ind_" + str(int(lr * 100)) + ".pkl"), "wb") as f:
        pickle.dump(acc_ind, f)


for lr in label_rates:
    experiment(lr)
# jl.Parallel(n_jobs=4)(jl.delayed(experiment)(lr) for lr in label_rates)
