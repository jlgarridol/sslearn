import os

import numpy as np
import pandas as pd


def save_keel(X, y, route, name=None, attribute_name=None, target_name="Class",  classification=True, unlabeled=True, force_targets=None):
    """Save a dataset in the KEEL format

    Parameters
    ----------
    X : array-like
        Dataset features
    y : array-like
        Dataset targets
    route : str
        Path to save the dataset
    name : str, optional
        Dataset name, if None the route basename will be selected, by default None
    attribute_name : list, optional
        List of attribute names, if None the default names will be used, by default None
    target_name : str, optional
        Target name, by default "Class"
    classification : bool, optional
        If the dataset is classification or regression, by default True
    unlabeled : bool, optional
        If the dataset has unlabeled instances, by default True
    force_targets : collection, optional
        Force the targets to be a specific value, by default None
    """    
    columns = []
    types = []
    min_max = []
    if name is None:
        name = os.path.basename(route).split(".")[0]

    unlabel_target = y == y.dtype.type(-1)
    if classification:
        y = y.astype("str")

        if unlabeled:            
            y = y.astype("str")
            if y.dtype.itemsize < np.array("unlabeled").dtype.itemsize:
                y = y.astype(f"<U{len('unlabeled')}")
            y[unlabel_target] = "unlabeled"
            if force_targets is not None:
                force_targets = force_targets.copy()
                force_targets.append("unlabeled")

    # Generate attributes:
    if attribute_name is None:
        if isinstance(X, pd.DataFrame):
            attribute_name = X.columns
        elif isinstance(X, np.ndarray):
            attribute_name = [f"a{i}" for i in range(X.shape[-1])]
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    data = pd.concat([X, pd.Series(y).rename(target_name)], axis=1)
    for i, col in enumerate(data):
        if i < len(attribute_name) and attribute_name[i] != col:
            columns.append(attribute_name[i])
        else:
            columns.append(col)
        numeric = False
        if data[col].dtype.kind in "ui":
            numeric = True
            types.append(" integer")
        elif data[col].dtype.kind == "f":
            numeric = True
            types.append(" real")
        elif data[col].dtype.kind in "bSOU":
            types.append("")
        if numeric:
            min_max.append(f" [{data[col].min()},{data[col].max()}]")
        else:
            if col == target_name and force_targets is not None:
                min_max.append(" {" + ",".join(force_targets) + "}")
            else:
                min_max.append(" {" + ",".join(data[col].unique()) + "}")

    # Generate header
    value = f"@relation {name}"
    for c, t, mm in zip(columns, types, min_max):
        value += f"\n@attribute {c}{t}{mm}"
    
    value += "\n@inputs " + ",".join(attribute_name)
    value += f"\n@outputs {target_name}"
    value += "\n@data\n"
    with open(route, "w") as f:
        f.write(value)
        data.to_csv(f, index=False, header=False)
