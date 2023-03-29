Semi-Supervised Learning Library (sslearn)
===

![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/jlgarridol/sslearn) ![Code Climate coverage](https://img.shields.io/codeclimate/coverage/jlgarridol/sslearn) ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/jlgarridol/sslearn/python-package.yml)

The `sslearn` library is a Python package for machine learning over Semi-supervised datasets. It is an extension of [scikit-learn](https://github.com/scikit-learn/scikit-learn).

Installation
---
### Dependencies

* scikit_learn = 1.2.0
* joblib = 1.2.0
* numpy = 1.23.3
* pandas = 1.4.3
* scipy = 1.9.3
* statsmodels = 0.13.2
* pytest = 7.2.0 (only for testing)

### `pip` installation

It can be installed using *Pypi*:

    pip install sslearn

Code example
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

Citing
---
```bibtex
@software{jose_luis_garrido_labrador_2023_7565222,
  author       = {José Luis Garrido-Labrador and
                  César García-Osorio and
                  Juan J. Rodríguez and
                  Jesus Maudes},
  title        = {jlgarridol/sslearn: V1.0.2},
  month        = feb,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {1.0.2},
  doi          = {10.5281/zenodo.7650049},
  url          = {https://doi.org/10.5281/zenodo.7650049}
}
```
