import pandas as pd
from arff2pandas import a2p
from ._preprocess import secure_dataset


def read_keel(path, format="pandas", secure=True, target_col=-1, **kwards):
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")
    with open(path, "r") as file:
        lines = file.readlines()
        counter = 1
        for _ in lines:
            counter += 1
            if "@data" in _:
                break
    data = pd.read_csv(path, skiprows=counter, header=None, **kwards)

    X = data.loc[:, data.columns != data.columns[target_col]]
    y = data.loc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y)
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y


def read_csv(path, format="pandas", secure=True, target_col=-1, **kwards):
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")
    data = pd.read_csv(path, **kwards)

    X = data.loc[:, data.columns != data.columns[target_col]]
    y = data.loc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y)
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y


def read_arff(path, format="pandas", secure=True, target_col=-1):
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")

    with open(path, "r") as file:
        data = a2p.load(file)

    X = data.loc[:, data.columns != data.columns[target_col]]
    y = data.loc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y)
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y
