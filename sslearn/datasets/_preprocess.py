import numpy as np


def secure_dataset(X, y):
    if np.issubdtype(y, np.number):
        y = y + 2

    return X, y
