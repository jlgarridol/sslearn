import pandas as pd
import warnings
from ._preprocess import secure_dataset

keel_type_cheat = {
    "string": "string",
    "integer": "int",
    "real": "float",
    "numeric": "double"
}


def read_keel(path, format="pandas", secure=False, target_col=None, encoding="utf-8", **kwards):
    """Read a .dat file from KEEL (http://www.keel.es/)

    Parameters
    ----------
    path : str
        File path
    format : str, optional
        Object that will contain the data, it can be `numpy` or `pandas`, by default "pandas"
    secure : bool, optional
        It guarantees that the dataset has not  `-1` as valid class, in order to make it semi-supervised after, by default False
    target_col : {str, int, None}, optional
        Column name or index to select class column, if None use the default value stored in the file, by default None
    encoding: str, optional
        Encoding of file, by default "utf-8"

    Returns
    -------
    X, y: array_like
        Dataset loaded.
    """
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")

    attributes = []
    types = []
    target = None
    with open(path, "r") as file:
        lines = file.readlines()
        counter = 1
        for line in lines:
            counter += 1
            if "@attribute" in line:
                parts = line.split(" ")
                name_ = parts[1]
                type_ = parts[2]
                if type_[0] == "{":
                    type_ = "string"
                attributes.append(name_)
                types.append(keel_type_cheat[type_])
            elif "@outputs" in line:
                target = line.split(" ")[1].strip('\n')
            elif "@data" in line:
                break
    if target is None:
        target = attributes[-1]
    data = pd.read_csv(path, skiprows=counter-1, header=None, **kwards)
    if len(data.columns) != len(attributes):
        warnings.warn(f"The dataset's have {len(data.columns)} columns but file declares {len(attributes)}.", RuntimeWarning)
        X = data
        y = None
    else:
        data.columns = attributes
        data = data.astype(dict(zip(attributes, types)))
        for att, tp in zip(attributes, types):
            if tp == "string":
                data[att] = data[att].str.strip()
        if target_col is None:
            target_col = target
        elif isinstance(target_col, int):
            target_col = data.columns[target_col]

        att_columns = attributes.copy()
        att_columns.remove(target_col)

        X = data[att_columns]
        y = data[target_col]

        y[y == "unlabeled"] = y.dtype.type(-1)
        if secure:
            X, y = secure_dataset(X, y)

    if format == "numpy":
        X = X.to_numpy().astype(float)
        y = y.to_numpy()
        if y.dtype == object:
            y = y.astype("str")
    return X, y


def read_csv(path, format="pandas", secure=False, target_col=-1, **kwards):
    """Read a .csv file

    Parameters
    ----------
    path : str
        File path
    format : str, optional
        Object that will contain the data, it can be `numpy` or `pandas`, by default "pandas"
    secure : bool, optional
        It guarantees that the dataset has not  `-1` as valid class, in order to make it semi-supervised after, by default False
    target_col : {str, int, None}, optional
        Column name or index to select class column, if None use the default value stored in the file, by default None

    Returns
    -------
    X, y: array_like
        Dataset loaded.
    """
    if format not in ["pandas", "numpy"]:
        raise AttributeError("Formats allowed are `pandas` or `numpy`")
    data = pd.read_csv(path, **kwards)

    if target_col is None:
        raise AttributeError("`read_csv` do not allow a `None` value for `target_col`, use `integer` or `string` instead.")
    elif isinstance(target_col, str):
        target_col = data.columns.index(target_col)

    X = data.iloc[:, data.columns != data.columns[target_col]]
    y = data.iloc[:, target_col]

    if secure:
        X, y = secure_dataset(X, y)
    if format == "numpy":
        X = X.to_numpy()
        y = y.to_numpy()
    return X, y