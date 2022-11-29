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
    if y.dtype.type(-1) in y:
        raise ValueError("The dataset contains -1 as valid class. Please, change it to another value.")
    return X, y
    # if np.issubdtype(y.dtype, np.number):
    #     y = y + 2

    # return X, y
