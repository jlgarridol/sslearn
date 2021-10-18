from sklearn.ensemble import RandomForestClassifier
from sslearn.supervised.rotation import RotationForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sktime.contrib.vector_classifiers._rotation_forest import RotationForest as RotationForestSktime
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier

import sslearn.wrapper as wrp 
from sslearn.model_selection import artificial_ssl_dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone as skclone
from sklearn.utils import check_random_state as crs
import colored
import warnings

import pickle
import gc 
import joblib as jl

import pandas as pd
import numpy as np
import os
import sys
import pathlib

save_path = pathlib.Path(__file__).parent.resolve()
path = "/home/jlgarridol/data/uci/csv"

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

to_delete = ["census", "kddcup", "fars", "poker", "connect-4", "adult"] # Quito los datasets grandes
for k in to_delete:
    if k in datasets:
        del datasets[k]

data_it = ['vowel', 'flare-solar', 'banana', 'autos', 'titanic', 'contraceptive', 'monks', 'thyroid', 'appendicitis', 'saheart', 'kr-vs-k', 'movement_libras', 'letter', 'hepatitis', 'segment', 'satimage', 'nursery', 'led7digit', 'chess', 'pima', 'abalone', 'car', 'shuttle', 'mammographic', 'lymphography', 'optdigits', 'penbased', 'glass', 'dermatology', 'sonar', 'bands', 'vehicle', 'iris', 'spectfheart', 'german', 'pagebloks', 'housevotes', 'haberman', 'australian', 'ecoli', 'texture', 'yeast', 'crx', 'balance', 'marketing', 'bupa', 'mushroom', 'tae', 'breast', 'tic-tac-toe', 'coil2000', 'spambase', 'wine', 'ionosphere', 'hayes-roth', 'phoneme', 'wdbc', 'splice', 'newthyroid', 'twonorm', 'zoo', 'ring', 'cleveland', 'post-operative', 'wisconsin', 'heart', 'magic']

seed = 100
classifier_seed = 0
repetitions = 5
global_rs = crs(seed)
label_rates = [x/10 for x in range(1,5)]
colors = ['red','blue','green','yellow','cyan','magenta','white','gray']

classifiers = {
    "DemocraticCoLearning": wrp.DemocraticCoLearning(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1)),
    "SelfTraining": wrp.SelfTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1)),
    "TriTraining": wrp.TriTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    "Rasco": wrp.Rasco(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    "RelRasco": wrp.RelRasco(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    "CoForest": wrp.CoForest(random_state=classifier_seed),
    "DeTriTraining": wrp.DeTriTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    "Setred": wrp.Setred(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    "CoTrainingByComitte": wrp.CoTrainingByCommittee(ensemble_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1), random_state=classifier_seed),
    "CoTraining": OneVsRestClassifier(wrp.CoTraining(base_estimator=RandomForestClassifier(random_state=classifier_seed, n_jobs=-1)))
}



def experiment(lr):
    #print("\nLabel rate: {}".format(int(lr*100)))
    warnings.filterwarnings("ignore")
    color_index=label_rates.index(lr)
    
    acc_trans, acc_ind = dict(), dict()
    for c in classifiers:
        acc_trans[c] = dict()
        acc_ind[c] = dict()
        for d in datasets:
            acc_trans[c][d] = list()
            acc_ind[c][d] = list()
    
    steps = len(classifiers)*len(datasets)*repetitions
    step = 0
    for i, d in enumerate(datasets):

        X, y = datasets[d]

        for c in classifiers:
            learner = skclone(classifiers[c])
            for r in range(repetitions):
                step += 1
                print(colored("Learning Rate {}, Classifier {}, Dataset {}, Repetition {}. Steps {}/{}".format(lr,c,d,r+1, step, steps),colors[color_index]))
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

    print("Finalizado",lr*100, file=sys.stderr)
    with open(save_path+"/acc_trans_"+str(int(lr*100)+".pkl"), "wb") as f:
        pickle.dump(acc_trans, f)
    with open(save_path+"/acc_ind_"+str(int(lr*100))+".pkl", "wb") as f:
        pickle.dump(acc_ind, f)


jl.Parallel(n_jobs=4)(jl.delayed(experiment)(lr) for lr in label_rates)