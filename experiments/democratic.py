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
    datasets[d] = list()
    for i in range(1, 11):
        data_path = os.path.join(path, f"{d}-ssl10")
        train = read_keel(
            os.path.join(data_path, f"{d}-ssl10-{i}tra.dat"), format="numpy"
        )
        trans = read_keel(
            os.path.join(data_path, f"{d}-ssl10-{i}trs.dat"), format="numpy"
        )
        test = read_keel(
            os.path.join(data_path, f"{d}-ssl10-{i}trs.dat"), format="numpy"
        )
        datasets[d].append((train, trans, test))

seed = 100
classifier_seed = 0

log.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
log.level("EVO", no=15)
log.add(
    "DemocraticCoLearning.log",
    format="{time} | {level} | {message}",
    filter="sslearn",
    level="EVO",
)

result = dict()


def experiment():
    print("Start experiments")
    warnings.filterwarnings("ignore")

    acc_trans, acc_ind = dict(), dict()
    c = "Democratic"
    acc_trans[c] = dict()
    acc_ind[c] = dict()
    for d in datasets:
        acc_trans[c][f"{d}-ssl10"] = list()
        acc_ind[c][f"{d}-ssl10"] = list()

    for d in data_it:
        print("Processing with", d)

        for i in range(10):
            (X_train, y_train), (X_trans, y_trans), (X_test, y_test) = datasets[d][i]
            learner = wrp.DemocraticCoLearning(
                base_estimator=[
                    DecisionTreeClassifier(
                        random_state=classifier_seed, min_samples_leaf=2
                    ),
                    GaussianNB(),
                    KNeighborsClassifier(n_neighbors=3),
                ],
                confidence_method="bernoulli",
                logging=True,
                log_name=f"{d}-ssl10-k{i}",
                save_dict=result,
            )

            learner.fit(X_train, y_train)
            score_trans = learner.score(X_trans, y_trans)
            score_ind = learner.score(X_test, y_test)

            acc_trans[c][f"{d}-ssl10"].append(score_trans)
            acc_ind[c][f"{d}-ssl10"].append(score_ind)

            del learner

    print("End experiment")


experiment()
with open("democratic_evolution_no_pow.pkl", "wb") as f:
    pk.dump(result, f)
