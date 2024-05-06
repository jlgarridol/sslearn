"""
Summary of module `sslearn.model_selection`:

This module contains functions to split datasets into training and testing sets.

## Functions

[artificial_ssl_dataset](#artificial_ssl_dataset):
> Generate an artificial semi-supervised learning dataset.

## Classes

[StratifiedKFoldSS](#StratifiedKFoldSS):
> Stratified K-Folds cross-validator for semi-supervised learning.


"""

from ._split import artificial_ssl_dataset, StratifiedKFoldSS

__all__ = ['artificial_ssl_dataset', 'StratifiedKFoldSS']