import pandas as pd
import numpy as np
from ._preprocess import secure_dataset

__data_keel_types = {
    "integer", np.dtype('int'),
    "real", np.dtype("double"),
    "numeric", np.dtype("double"),
    "string", np.dtype("|S"),
}


def read_keel(path, format="pandas", secure=True, unlabel=True, **kwards):
    """Read .dat file from Knowledge Extraction based on Evolutionary Learning (KEEL)
    Parameters
    ----------
    path : string
        File path
    format : str, optional
        The kind of data structure to load the file, may be `pandas` for DataFrame or `numpy` for array , by default "pandas"
    secure : bool, optional
        If `secure` is True then if exists a -1 value in target classes the target values will be increased in two values., by default True
    unlabel : bool, optional
        If `unlabel` is True then the class "unlabel" will be change to -1, by default True

    Returns
    -------
    {pandas Dataframe|numpy array}
        Dataset loaded
    """
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")
    types_ = []
    columns_ = []
    with open(path, "r") as file:
        lines = file.readlines()
        counter = 1
        for _ in lines:
            counter += 1
            if "@attribute" in _:
                splitted = _.split(" ")
                columns_.append(splitted[1])
                type_ = splitted[2]
                if type_[0] == "{":
                    type_ = "string"
            if "@data" in _:
                break
    data = pd.read_csv(path, skiprows=counter, header=None, **kwards)

    X = data.loc[:, data.columns != data.columns[target_col]]
    X.columns = []

    y = data.loc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y)
    if unlabel:
        y[y == "unlabel"] = -1
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y


def read_csv(path, format="pandas", secure=True, target_col=-1, **kwards):
    """Read .csv file.
    Parameters
    ----------
    path : string
        File path
    format : str, optional
        The kind of data structure to load the file, may be `pandas` for DataFrame or `numpy` for array , by default "pandas"
    secure : bool, optional
        If `secure` is True then if exists a -1 value in target classes the target values will be increased in two values., by default True
    target_col : int, optional
        Select the column to mark as target. If is -1 then the last column will be selected. , by default -1

    Returns
    -------
    {pandas Dataframe|numpy array}
        Dataset loaded
    """
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")
    data = pd.read_csv(path, **kwards)

    X = data.loc[:, data.columns != data.columns[target_col]]
    y = data.loc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y, target_column=target_col)
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y


def read_arff(path, format="pandas", secure=True, target_col=-1):
    """Read .arff file from WEKA. It requires `arff2pandas`
    Parameters
    ----------
    path : string
        File path
    format : str, optional
        The kind of data structure to load the file, may be `pandas` for DataFrame or `numpy` for array , by default "pandas"
    secure : bool, optional
        If `secure` is True then if exists a -1 value in target classes the target values will be increased in two values., by default True
    target_col : int, optional
        Select the column to mark as target. If is -1 then the last column will be selected. , by default -1

    Returns
    -------
    {pandas Dataframe|numpy array}
        Dataset loaded
    """
    from arff2pandas import a2p

    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")

    with open(path, "r") as file:
        data = a2p.load(file)

    X = data.loc[:, data.columns != data.columns[target_col]]
    y = data.loc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y, target_column=target_col)
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y
