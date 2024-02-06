import sklearn.model_selection as ms
from sklearn.utils import check_random_state
import numpy as np


class StratifiedKFoldSS():
    def __init__(self, n_splits=5, shuffle=False, random_state=None):

        self.K = ms.StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                                    random_state=random_state)
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """Generate a artificial dataset based on StratifiedKFold method

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        Yields
        -------
        X : ndarray
            The feature set.
        y : ndarray
            The label set, -1 for unlabel instance.
        label: ndarray
            The training set indices for split mark as labeled.
        unlabel: ndarray
            The training set indices for split mark as unlabeled.
        """
        for train, test in self.K.split(X, y):
            # Inverse train and test because train is big dataset
            label = test
            unlabel = train

            X_label, y_label, X_unlabel, y_unlabel = X[label], y[label],\
                X[unlabel], np.array([-1] * len(unlabel))
            X_ = np.concatenate((X_label, X_unlabel), axis=0)
            y_ = np.concatenate((y_label, y_unlabel), axis=0)

            yield X_, y_, label, unlabel


def artificial_ssl_dataset(X, y, label_rate=0.1, random_state=None, force_minimum=None, indexes=False, **kwards):
    """Create an artificial Semi-supervised dataset from a supervised dataset.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.
    label_rate : float, optional
        Proportion between labeled instances and unlabel instances, by default 0.1
    random_state : int or RandomState, optional
        Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls, by default None
    force_minimum: int, optional
        Force a minimum of instances of each class, by default None
    indexes: bool, optional
        If True, return the indexes of the labeled and unlabeled instances, by default False
    shuffle: bool, default=True
        Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
    stratify: array-like, default=None
        If not None, data is split in a stratified fashion, using this as the class labels.

    Returns
    -------
    X : ndarray
        The feature set.
    y : ndarray
        The label set, -1 for unlabel instance.
    X_unlabel: ndarray
        The feature set for each y mark as unlabel
    y_unlabel: ndarray
        The true label for each y in the same order.
    label: ndarray (optional)
        The training set indexes for split mark as labeled.
    unlabel: ndarray (optional)
        The training set indexes for split mark as unlabeled.
    """
    assert (label_rate > 0) and (label_rate < 1),\
        "Label rate must be in (0, 1)."
    assert "test_size" not in kwards and "train_size" not in kwards,\
        "Test size and train size are illegal parameters in this method."

    indices = np.arange(len(y))

    if force_minimum is not None:
        try:
            selected = __random_select_n_instances(y, force_minimum, random_state)
        except ValueError:
            raise ValueError("The number of instances of each class is less than force_minimum.")

        # Remove selected instances from indices
        indices = np.delete(indices, selected, axis=0)    

    # Train test split with indexes
    label, unlabel = ms.train_test_split(indices, train_size=label_rate,
                                         random_state=random_state, **kwards)

    if force_minimum is not None:
        label = np.concatenate((selected, label))
    
    # Create the label and unlabel sets
    X_label, y_label, X_unlabel, y_unlabel = X[label], y[label],\
        X[unlabel], np.array([-1] * len(unlabel))

    # Create the artificial dataset
    X = np.concatenate((X_label, X_unlabel), axis=0)
    y = np.concatenate((y_label, y_unlabel), axis=0)

    if indexes:
        return X, y, X_unlabel, y_unlabel, label, unlabel

    return X, y, X_unlabel, y_unlabel


    """    
    if force_minimum is not None:
        try:
            selected = __random_select_n_instances(y, force_minimum, random_state)
        except ValueError:
            raise ValueError("The number of instances of each class is less than force_minimum.")
        X_selected = X[selected]
        y_selected = y[selected]

        # Remove selected instances from X and y
        X = np.delete(X, selected, axis=0)
        y = np.delete(y, selected, axis=0)
    
    X_label, X_unlabel, y_label, true_label = \
        ms.train_test_split(X, y,
                            train_size=label_rate,
                            random_state=random_state, **kwards)
    X = np.concatenate((X_label, X_unlabel), axis=0)
    y = np.concatenate((y_label, np.array([-1] * len(true_label))), axis=0)

    if force_minimum is not None:
        X = np.concatenate((X, X_selected), axis=0)
        y = np.concatenate((y, y_selected), axis=0)
    
    if indexes:
        return X, y, X_unlabel, true_label, X_label, X_unlabel

    return X, y, X_unlabel, true_label
    """

def __random_select_n_instances(y, n, random_state):

    # Select n instances of each class randomly
    classes = np.unique(y)
    selected = []
    random_state = check_random_state(random_state)
    for c in classes:
        idx = np.where(y == c)[0]
        selected.append(random_state.choice(idx, n, replace=False))
    selected = np.concatenate(selected)
    return selected