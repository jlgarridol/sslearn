"""
Summary of module `sslearn.datasets`:

This module contains functions to load and save datasets in different formats.

## Functions

1. read_csv : Load a dataset from a CSV file.
2. read_keel : Load a dataset from a KEEL file.
3. secure_dataset : Secure the dataset by converting it into a secure format.
4. save_keel : Save a dataset in KEEL format.


"""

from ._loader import read_csv, read_keel
from ._writer import save_keel
from ._preprocess import secure_dataset

__all__ = ["read_csv", "read_keel", "secure_dataset", "save_keel"]