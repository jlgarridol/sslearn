import gc
import os
import pickle
import sys
import joblib as jl
import numpy as np
import pandas as pd
import pathlib
import warnings
from sklearn.base import clone as skclone
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state as crs
from termcolor import colored

sys.path.insert(1, "..")
from sslearn.model_selection import artificial_ssl_dataset
import sslearn.wrapper as wrp

save_path = pathlib.Path(__file__).parent.resolve()
path = "/home/jlgarridol/DATA/uci/classification/csv"

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
global_rs = crs(seed)
label_rates = [x / 10 for x in range(1, 5)]
colors = ["red", "blue", "green", "yellow", "cyan", "magenta", "white", "gray"]

classifiers = {
    "DemocraticCoLearning": wrp.DemocraticCoLearning(base_estimator=[DecisionTreeClassifier(random_state=classifier_seed, min_samples_leaf=2),
                                                                     GaussianNB(),
                                                                     KNeighborsClassifier(n_neighbors=3)]),
    "TriTraining": wrp.TriTraining(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), random_state=classifier_seed),
    "CoTrainingByComitte": wrp.CoTrainingByCommittee(ensemble_estimator=BaggingClassifier(n_estimators=3, random_state=classifier_seed),
                                                     poolsize=100,
                                                     max_iterations=40,
                                                     random_state=classifier_seed),
}


def experiment(lr):
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
            learner = skclone(classifiers[c])
            for r in range(repetitions):
                step += 1

                skf = StratifiedKFold(
                    n_splits=n_splits, random_state=seed, shuffle=True
                )

                # cols = curses.tigetnum('cols')
                text = "Learning Rate {}, Classifier {}, Dataset {}, Repetition {}. Steps {}/{}".format(
                    lr, c, d, r + 1, step, steps
                )
                # points = cols-len(text)-len(vueltas)
                final_text = f"{text}"
                print(colored(final_text, colors[color_index]))
                for train, test in skf.split(X, y):

                    score_trans, score_ind = list(), list()
                    X_train, y_train = X[train], y[train]
                    X_test, y_test = X[test], y[test]

                    X_, y_, X_unlabel, y_true = artificial_ssl_dataset(
                        X_train,
                        y_train,
                        label_rate=lr,
                        random_state=global_rs.randint(100),
                    )
                    learner.random_state = classifier_seed + r

                    # If allow unlabel examples
                    learner.fit(X_, y_)
                    # learner.fit(X_[y_ != y_.dtype.type(-1)], y_[y_ != y_.dtype.type(-1)])

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
    with open(os.path.join(save_path, "acc_trans_" + str(int(lr * 100)) + ".pkl"), "wb") as f:
        pickle.dump(acc_trans, f)
    with open(os.path.join(save_path, "acc_ind_" + str(int(lr * 100)) + ".pkl"), "wb") as f:
        pickle.dump(acc_ind, f)


# for lr in label_rates:
#     experiment(lr)
jl.Parallel(n_jobs=4)(jl.delayed(experiment)(lr) for lr in label_rates)
