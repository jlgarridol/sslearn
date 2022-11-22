import numpy as np
import pandas as pd
import os
import sys
import warnings
from sklearn.base import clone as skclone
from sklearn.ensemble import BaggingClassifier
import re

sys.path.insert(1, "..")
from sslearn.base import FakedProbaClassifier
from sslearn.datasets import read_keel
import sslearn.wrapper as wrp
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle as pk
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier, OutputCodeClassifier
from sklearn.calibration import CalibratedClassifierCV

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
    # "letter",
    # "lymphography",
    # "magic",
    # "mammographic",
    "marketing",
    # "monks",
    # "movement_libras",
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
    # "thyroid",
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

path = os.path.join(os.path.dirname(__file__), "..", "data")
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
            os.path.join(data_path, f"{d}-ssl10-{i}tst.dat"), format="numpy"
        )
        datasets[d].append((train, trans, test))

seed = 100
classifier_seed = 0

# log.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")
# log.level("EVO", no=15)
# log.add(
#     "DemocraticCoLearning.log",
#     format="{time} | {level} | {message}",
#     filter="sslearn",
#     level="EVO",
# )

base = {
    "DecisionTree": DecisionTreeClassifier(
        random_state=classifier_seed, min_samples_leaf=2
    ),
    "NaiveBayes": GaussianNB(),
    "3NN": KNeighborsClassifier(n_neighbors=3)
}

base["Bagging"] = {"n_estimators": 100, "random_state": classifier_seed, "n_jobs": 10}

learners = {
    # "Democratic": wrp.DemocraticCoLearning(
    #     base_estimator=[
    #         skclone(base["DecisionTree"]),
    #         skclone(base["NaiveBayes"]),
    #         skclone(base["3NN"])
    #     ],
    #     confidence_method="bernoulli"
    # ),
    # "DemocraticOVO": wrp.DemocraticCoLearning(
    #     base_estimator=[
    #         CalibratedClassifierCV(OneVsOneClassifier(skclone(base["DecisionTree"]), n_jobs=10)),
    #         CalibratedClassifierCV(OneVsOneClassifier(skclone(base["NaiveBayes"]), n_jobs=10)),
    #         CalibratedClassifierCV(OneVsOneClassifier(skclone(base["3NN"]), n_jobs=10))
    #     ],
    #     confidence_method="bernoulli"
    # ),
    # "DemocraticOVR": wrp.DemocraticCoLearning(
    #     base_estimator=[
    #         OneVsRestClassifier(skclone(base["DecisionTree"]), n_jobs=10),
    #         OneVsRestClassifier(skclone(base["NaiveBayes"]), n_jobs=10),
    #         OneVsRestClassifier(skclone(base["3NN"]), n_jobs=10)
    #     ],
    #     confidence_method="bernoulli"
    # ),
    # "DemocraticECOC": wrp.DemocraticCoLearning(
    #     base_estimator=[
    #         OutputCodeClassifier(skclone(base["DecisionTree"]), n_jobs=10, random_state=classifier_seed),
    #         OutputCodeClassifier(skclone(base["NaiveBayes"]), n_jobs=10, random_state=classifier_seed),
    #         OutputCodeClassifier(skclone(base["3NN"]), n_jobs=10, random_state=classifier_seed)
    #     ],
    #     confidence_method="bernoulli"
    # ),
    "TriTraining": wrp.TriTraining(
        base_estimator=skclone(base["DecisionTree"]),
        random_state=classifier_seed,
        n_jobs=3
    ),
    "TriTrainingOVO": wrp.TriTraining(
        base_estimator=CalibratedClassifierCV(OneVsOneClassifier(skclone(base["DecisionTree"]), n_jobs=10)),
        random_state=classifier_seed,
        n_jobs=3
    ),
    "TriTrainingOVR": wrp.TriTraining(
        base_estimator=OneVsRestClassifier(skclone(base["DecisionTree"]), n_jobs=10),
        random_state=classifier_seed,
        n_jobs=3
    ),
    "TriTrainingECOC": wrp.TriTraining(
        base_estimator=OutputCodeClassifier(skclone(base["DecisionTree"]), n_jobs=10, random_state=classifier_seed),
        random_state=classifier_seed,
        n_jobs=3
    ),
    # "CoTrainingByCommitteeOVO": wrp.CoTrainingByCommittee(
    #     ensemble_estimator=BaggingClassifier(CalibratedClassifierCV(OneVsOneClassifier(skclone(base["DecisionTree"]), n_jobs=10)), **base["Bagging"]),
    #     random_state=classifier_seed
    # ),
    # "CoTrainingByCommitteeOVR": wrp.CoTrainingByCommittee(
    #     ensemble_estimator=BaggingClassifier(OneVsRestClassifier(skclone(base["DecisionTree"]), n_jobs=10), **base["Bagging"]),
    #     random_state=classifier_seed
    # ),
    # "CoTrainingByCommitteeECOC": wrp.CoTrainingByCommittee(
    #     ensemble_estimator=BaggingClassifier(OutputCodeClassifier(skclone(base["DecisionTree"]), n_jobs=10, random_state=classifier_seed), **base["Bagging"]),
    #     random_state=classifier_seed
    # ),
    # "CoForest": wrp.CoForest(skclone(base["DecisionTree"]), random_state=classifier_seed),
    # "CoForestOVO": wrp.CoForest(CalibratedClassifierCV(OneVsOneClassifier(skclone(base["DecisionTree"]), n_jobs=10)), random_state=classifier_seed),
    # "CoForestOVR": wrp.CoForest(OneVsRestClassifier(skclone(base["DecisionTree"]), n_jobs=10), random_state=classifier_seed),
    # "CoForestECOC": wrp.CoForest(FakedProbaClassifier(OutputCodeClassifier(skclone(base["DecisionTree"]), n_jobs=10, random_state=classifier_seed)), random_state=classifier_seed),
    # "Setred": wrp.Setred(skclone(base["3NN"]), random_state=classifier_seed, n_jobs=3),
    # "SetredOVO": wrp.Setred(CalibratedClassifierCV(OneVsOneClassifier(skclone(base["3NN"]), n_jobs=10)), random_state=classifier_seed, n_jobs=3),
    # "SetredOVR": wrp.Setred(OneVsRestClassifier(skclone(base["3NN"]), n_jobs=10), random_state=classifier_seed, n_jobs=3),
    # "SetredECOC": wrp.Setred(FakedProbaClassifier(OutputCodeClassifier(skclone(base["3NN"]), n_jobs=10, random_state=classifier_seed)), random_state=classifier_seed, n_jobs=3),
    # "SelfTraining": wrp.SelfTraining(skclone(base["DecisionTree"])),
    # "SelfTrainingOVO": wrp.SelfTraining(CalibratedClassifierCV(OneVsOneClassifier(skclone(base["DecisionTree"]), n_jobs=10))),
    # "SelfTrainingOVR": wrp.SelfTraining(OneVsRestClassifier(skclone(base["DecisionTree"]), n_jobs=10)),
    # "SelfTrainingECOC": wrp.SelfTraining(FakeProbaClassifier(OutputCodeClassifier(skclone(base["DecisionTree"]), n_jobs=10, random_state=classifier_seed))),
}

trained_learners = dict()

def experiment():
    print("Start experiments")
    warnings.filterwarnings('ignore', category=Warning,
                            module=r'^{0}\.'.format(re.escape(__name__)))

    acc_trans, acc_ind = dict(), dict()
    for c in learners:
        acc_trans[c] = dict()
        acc_ind[c] = dict()
        for d in datasets:
            acc_trans[c][f"{d}-ssl10"] = list()
            acc_ind[c][f"{d}-ssl10"] = list()

    for d in data_it:
        print("Processing with", d)

        for c in learners:
            learner = skclone(learners[c])
        
            for i in range(10):
                (X_train, y_train), (X_trans, y_trans), (X_test, y_test) = datasets[d][i]
                
                learner.fit(X_train, y_train)
                score_trans = learner.score(X_trans, y_trans)
                score_ind = learner.score(X_test, y_test)

                acc_trans[c][f"{d}-ssl10"].append(score_trans)
                acc_ind[c][f"{d}-ssl10"].append(score_ind)

                trained_learners[f"{c}-k{i}-{d}-ssl10"] = learner

    with open("binary_second_trans.pkl", "wb") as f:
        pk.dump(acc_trans, f)
    with open("binary_second_ind.pkl", "wb") as f:
        pk.dump(acc_ind, f)
    with open("binary_second_models.pkl", "wb") as f:
        pk.dump(trained_learners, f)
    print("End experiment")


experiment()
