Semi-Supervised Learning Library (sslearn)
===

![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/jlgarridol/sslearn) ![Code Climate coverage](https://img.shields.io/codeclimate/coverage/jlgarridol/sslearn) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/jlgarridol/sslearn/Python%20package)

The `sslearn` library is a Python package for machine learning over Semi-supervised datasets. It is an extension of [scikit-learn](https://github.com/scikit-learn/scikit-learn).

Installation
---
### Dependencies

* scikit_learn = 1.1.2
* joblib = 1.1.0
* numpy = 1.23.3
* pandas = 1.4.3
* scipy = 1.9.3
* statsmodels = 0.13.2
* pytest = 7.2.0 (only for testing)

### `pip` installation

It can be installed using *Pypi*:

    pip install sslearn

How to
---
```python
from sslearn.wrapper import TriTraining
from sslearn.model_selection import artificial_ssl_dataset
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X, y, X_unlabel, true_label = artificial_ssl_dataset(X, y, label_rate=0.1)

model = TriTraining().fit(X, y)
model.score(X_unlabel, true_label)
```
