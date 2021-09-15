import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
patch_sklearn()

import os, gc, sys, pickle
sys.path.insert(1,"..")

from sslearn.wrapper import SelfTraining, Rasco, RelRasco, RotRelRasco
from sslearn.supervised.rotation import RotationForestClassifier, RotatedTree
from sslearn.model_selection import artificial_ssl_dataset

from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone as skclone
from sklearn.utils import check_random_state as crs

import warnings
warnings.filterwarnings("ignore")

path = "/route/to/datasets"

seed = 100
classifier_seed = 0
repetitions = 5
global_rs = crs(seed)
label_rates = [0.1, 0.2, 0.3, 0.4]

classifiers = {
    "Rot-Rel-Rasco_batch": RotRelRasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), 
                                       n_estimators=30, random_state=classifier_seed, incremental=False),
    "Rot-Rel-Rasco_incremental": RotRelRasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), 
                                             n_estimators=30, random_state=classifier_seed, incremental=True),
    "Rot-Rel-Rasco_batch_pre": RotRelRasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), 
                                           n_estimators=30, random_state=classifier_seed, incremental=False, pre_rotation=True),
    "Rot-Rel-Rasco_incremental_pre": RotRelRasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), 
                                                 n_estimators=30, random_state=classifier_seed, incremental=True, pre_rotation=True),
    "Rasco_batch": Rasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), 
                         n_estimators=30, random_state=classifier_seed, incremental=False),
    "Rasco_incremental": Rasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed),
                         n_estimators=30, random_state=classifier_seed, incremental=True),
    "Rel-Rasco_batch": RelRasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed),
                                n_estimators=30, random_state=classifier_seed, incremental=False),
    "Rel-Rasco_incremental": RelRasco(base_estimator=DecisionTreeClassifier(random_state=classifier_seed),
                                      n_estimators=30, random_state=classifier_seed, incremental=True),
    "SelfTraining": SelfTraining(base_estimator=RotationForestClassifier(base_estimator=DecisionTreeClassifier(random_state=classifier_seed), 
                                 n_estimators=10, random_state=classifier_seed), max_iter=40)
}

datasets = {}
for file in os.listdir(path):
    dataset = pd.read_csv(os.path.join(path,file), header=None)

    columns = []
    for i, tp in enumerate(dataset.dtypes):
        if not np.issubdtype(tp, np.number) and i != dataset.shape[1]-1:
            columns.append(i)
            dataset[i] = dataset[i].astype("|S")
    
    y = dataset.iloc[:,-1]
    if np.issubdtype(y, np.number):
        y = y+2 
    X = dataset.iloc[:,:-1]
    if len(columns) > 0:
        elements = [X[X.columns.difference(columns)]]

        for col in columns:
            elements.append(pd.get_dummies(X[col]))
        
        concatenated_data = pd.concat(elements, axis=1)
        X = concatenated_data

    
    
    datasets[file.split(".")[0]] = (X.to_numpy(), y.to_numpy())

for lr in label_rates:

    acc_trans, acc_ind = dict(), dict()
    for c in classifiers:
        acc_trans[c] = dict()
        acc_ind[c] = dict()
        for d in datasets:
            acc_trans[c][d] = list()
            acc_ind[c][d] = list()

    for i, d in enumerate(datasets):
        gc.collect()

        X, y = datasets[d]

        for c in classifiers:
            learner = classifiers[c]
            for r in range(repetitions):
                skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
                for train, test in skf.split(X, y):
                    score_trans, score_ind = list(), list()
                    X_train, y_train = X[train], y[train]
                    X_test, y_test = X[test], y[test]

                    X_, y_, X_unlabel,  y_true = artificial_ssl_dataset(X_train, y_train, label_rate=lr, random_state=global_rs.randint(100))
                    learner.random_state = classifier_seed+r
                    learner.fit(X_, y_)

                    score_trans = learner.score(X_unlabel, y_true)
                    score_ind = learner.score(X_test, y_test)

                    acc_trans[c][d].append(score_trans)
                    acc_ind[c][d].append(score_ind)
    with open("acc_trans_"+str(int(lr*10)), "wb") as f:
        pickle.dump(acc_trans, f)
    with open("acc_ind_"+str(int(lr*10)), "wb") as f:
        pickle.dump(acc_ind, f)






