Semi-Supervised Learning Library (sslearn)
===

<!-- Insert logo in the middle -->
<img width="100%" src="https://raw.githubusercontent.com/jlgarridol/sslearn/main/docs/sslearn.webp"/>

![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/jlgarridol/sslearn) ![Code Climate coverage](https://img.shields.io/codeclimate/coverage/jlgarridol/sslearn) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/jlgarridol/sslearn/python-package.yml) ![PyPI - Version](https://img.shields.io/pypi/v/sslearn) [![Static Badge](https://img.shields.io/badge/doc-available-blue?style=flat)](https://jlgarridol.github.io/sslearn/)

The `sslearn` library is a Python package for machine learning over Semi-supervised datasets. It is an extension of [scikit-learn](https://github.com/scikit-learn/scikit-learn).

## Installation


### Dependencies

* joblib >= 1.2.0
* numpy >= 1.23.3
* pandas >= 1.4.3
* scikit_learn >= 1.2.0
* scipy >= 1.10.1
* statsmodels >= 0.13.2
* pytest = 7.2.0 (only for testing)

### `pip` installation

It can be installed using *Pypi*:

    pip install sslearn

## Code example


```python
from sslearn.wrapper import TriTraining
from sslearn.model_selection import artificial_ssl_dataset
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X, y, X_unlabel, true_label = artificial_ssl_dataset(X, y, label_rate=0.1)

model = TriTraining().fit(X, y)
model.score(X_unlabel, true_label)
```

## Citing 

```bibtex
@software{garrido2024sslearn,
  author       = {José Luis Garrido-Labrador},
  title        = {jlgarridol/sslearn},
  month        = feb,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7565221},
}
```
