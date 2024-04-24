"""
Summary of module `sslearn.subview`:

This module contains classes to train a classifier or a regressor selecting a sub-view of the data.

Classes
-------
SubViewClassifier : Train a sub-view classifier.
SubViewRegressor : Train a sub-view regressor.

All doc
-------
"""

from ._subview import SubViewClassifier, SubViewRegressor

__all__ = ["SubViewClassifier", "SubViewRegressor"]