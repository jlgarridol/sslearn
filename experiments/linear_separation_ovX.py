import numpy as np
import pandas as pd
import os
import sys
import warnings
from sklearn.base import clone as skclone
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier


sys.path.insert(1, "..")
from sslearn.datasets import read_keel
import sslearn.wrapper as wrp
from sklearn.svm import LinearSVC
from loguru import logger as log
import pickle as pk

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

path = "/home/jlgarridol/Dropbox/GitHub/sslearn/data"
datasets = {}
for d in data_it:
    data_path = os.path.join(path, f"{d}-ssl10")
    train = read_keel(
        os.path.join(data_path, f"{d}-ssl10-1tra.dat"), format="numpy"
    )
    trans = read_keel(
        os.path.join(data_path, f"{d}-ssl10-1trs.dat"), format="numpy"
    )
    test = read_keel(
        os.path.join(data_path, f"{d}-ssl10-1trs.dat"), format="numpy"
    )
    X = np.concatenate((train[0][train[1] != train[1].dtype.type(-1)], trans[0], test[0]))
    y = np.concatenate((train[1][train[1] != train[1].dtype.type(-1)], trans[1], test[1]))
    datasets[d] = (X, y)

seed = 100
classifier_seed = 0


def experiment():
    print("Start experiments")
    warnings.filterwarnings("ignore")

    acc_ovo = dict()
    acc_ovr = dict()
    for d in datasets:
        acc_ovo[d] = list()
        acc_ovr[d] = list()

    for d in data_it:
        print("Processing with", d)

        X, y = datasets[d]
        learner = OneVsOneClassifier(LinearSVC(C=1e6, max_iter=10000), n_jobs=-1)
        learner.fit(X, y)

        acc_ovo[d].append(learner.score(X, y))

        learner2 = OneVsRestClassifier(LinearSVC(C=1e6, max_iter=10000), n_jobs=-1)
        learner2.fit(X, y)

        acc_ovr[d].append(learner2.score(X, y))

        del learner
        del learner2
    print("End experiment")
    return acc_ovo, acc_ovr


acc_ovo, acc_ovr = experiment()

with open("linear_separation_ovo.pkl", "wb") as f:
    pk.dump(acc_ovo, f)

with open("linear_separation_ovr.pkl", "wb") as f:
    pk.dump(acc_ovr, f)
