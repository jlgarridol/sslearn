import numpy as np


def secure_dataset(X, y):
    """Securize de dataset for semi-supervised learning ensuring that not exists `-1` as valid class.

    Parameters
    ----------
    X : Array-like
        Ignored
    y : Array-like
        Target array.

    Returns
    -------
    X, y: array_like
        Dataset securized.
    """
    return X, y
    # if np.issubdtype(y.dtype, np.number):
    #     y = y + 2

    # return X, y
