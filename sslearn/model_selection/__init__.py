"""
Summary of module `sslearn.model_selection`:

This module contains functions to split datasets into training and testing sets.

Functions
---------
artificial_ssl_dataset : Generate an artificial semi-supervised learning dataset.

Classes
-------
StratifiedKFoldSS : Stratified K-Folds cross-validator for semi-supervised learning.

All doc
----
"""

from ._split import *

__all__ = ['StratifiedKFoldSS', 'artificial_ssl_dataset']