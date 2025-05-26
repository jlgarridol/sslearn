---

---

Examples
=============

In this page we provide some code examples to show how to use the library. Also, exists a [Jupyter Notebook](https://colab.research.google.com/drive/1wKSz-f7N4elqQwz_phrWXDrf3lRqaD6s#scrollTo=KS-6GNxCayxf) with the same examples to run in Google Colab.


## Datasets manipulation

SSLearn include tools for loading csv and dat (KEEL) datasets. Also, it provides a function to generate a semi-supervised dataset from a labeled dataset.

### Load a CSV or a DAT (KEEL)  dataset

```python
from sslearn.datasets import read_csv, read_keel

# The CSV file must have the target column in the last position by default.
X_iris, y_iris = read_csv('iris.csv', format='numpy') # Format can be 'numpy' or 'pandas'

# The DAT file must have the target column in the first position by default. Also, the instances with class "unlabeled" will change to -1.
X_iris, y_iris = read_keel('iris.dat', format='numpy')
```

### Generate a unsupervised dataset from a labeled dataset

To test the semi-supervised algorithms, we need to generate a dataset with some instances without labels. The library provides a function to do this. 

Exists two ways to generate the dataset:
1. Randomly select some instances to be unlabeled giving a percentage of labeled instances.
2. Generate folds where one fold is the labeled dataset and the others are unlabeled.

```python
from sslearn.model_selection import artificial_ssl_dataset

# Using the previous dataset "X_iris" and "y_iris"
X_train, y_train, X_u, y_y = artificial_ssl_dataset(X_iris, y_iris, label_rate=0.2, random_state=42) # 80% of the instances are unlabeled
# or if you want also get the indexes of the instances
X_train, y_train, X_u, y_y, idx_label, idx_u = artificial_ssl_dataset(X_iris, y_iris, label_rate=0.2, random_state=42, indexes=True)
```
`artificial_ssl_dataset` supports stratify and shuffle as it works in `train_test_split` from `sklearn.model_selection`. Also, a minimum number of instances per class can be set with the parameter `force_minimum`.

The other way to generate the dataset is using the `StratifiedKFoldSS` class.

```python
from sslearn.model_selection import StratifiedKFoldSS

skss = StratifiedKFoldSS(n_splits=10, shuffle=True, random_state=42) # 10 folds, that implies 10% of the instances are labeled

for X_train, y_train, label_indices, unlabel_indices in skss.split(X_iris, y_iris):
    # It works like the StratifiedKFold from sklearn.model_selection
    pass # Here you can use the labeled and unlabeled instances
```


`X_train` and `y_train` are the set ready to be used in a semi-supervised algorithm, with the "-1" in the target column for the unlabeled instances.

## Wrappers

The wrappers are the most widely used algorithms in the semi-supervised learning field. The library includes the most popular ones. The algorithms included are:
* `SelfTraining`: Self-training algorithm for one classifier.
* `Setred`: Self-training with data amendment.
* `CoTraining`: The co-training algorithm for two views, supports multi-learning and requires two views.
* `Rasco`: Random subspace co-training, supports multi-learning.
* `RelRasco`: Relevant random subspace co-training, supports multi-learning.
* `DemocraticCoLearning`: Democratic co-learning algorithm, only for multi-learning.
* `CoForest`: Co-training version of Random Forest.
* `CoTrainingByCommittee`: Co-training by committee algorithm, a.k.a. CoBagging or CoBoosting depending on the base classifier in the literature.
* `TriTraining`: Tri-training algorithm, supports multi-learning.
* `DeTriTraining`: Tri-training with data amendment, supports multi-learning.

The wrappers follow the same structure as the classifiers in scikit-learn. Use `fit` to train the model and `predict` to predict the target column. Also, the `score` method is available to get the accuracy of the model.

```python
# Using the previous dataset "X_train" and "y_train" 

from sslearn.wrapper import CoForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn

# Load dataset
X_iris_train, y_iris_train, X_iris_U, y_iris_U = artificial_ssl_dataset(X_iris, y_iris, label_rate=0.1, random_state=42)

# Create the wrapper
coforest = CoForest(base_estimator=DecisionTreeClassifier(random_state=10), random_state=10)
# Train
coforest.fir(X_iris_train, y_iris_train)
# Predict
y_pred = coforest.predict(X_iris_U)
# Score
accuracy = accuracy_score(y_iris_U, y_pred)
```
Each wrapper has its parameters, some of them need different views, have specific parameters, or need a kind of base classifiers. The documentation of each wrapper explains the parameters and the requirements and provides examples of how to use them.

The wrappers that supports multi-learning can recive differents algorithms as base classifiers. The base classifiers must be a list of classifiers. 

```python
from sslearn.wrapper import DemocraticCoLearning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load dataset
X_iris_train, y_iris_train, X_iris_U, y_iris_U = artificial_ssl_dataset(X_iris, y_iris, label_rate=0.1, random_state=42)

# Create the wrapper
democratic = DemocraticCoLearning(base_estimator=[DecisionTreeClassifier(random_state=10), KNeighborsClassifier(), GaussianNB()], random_state=10)
# Train
democratic.fit(X_iris_train, y_iris_train)
```


### The multiview adapters
`CoTraining` is the only wrapper that supports different views. But, the library provides an adapter to provide a different view to the other wrappers. The adapters are in the `sslearn.subview` module. Supports classification and regression problems.

That adapter create a subview for the dataset, and when the wrapper train the base classifiers, each will be trained with the view defined in the adapter. It supports three modes of create subviews: 
- `regex`: The columns that match the regex will be in the subview.
- `columns`: The columns that are in the list will be in the subview.
- `include`: The columns whose names are in the list will be in the subview. 

```python

from sslearn.subview import SubViewClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset 
X_breast, y_breast = read_csv('breast-cancer.csv', format='numpy')
X_breast_train, y_breast_train, X_breast_U, y_breast_U = artificial_ssl_dataset(X_breast, y_breast, label_rate=0.1, random_state=42) 
# Create the the adapter
subview1 = SubViewClassifier(DecisionTreeClassifier(random_state=10), "breast", mode='include') # The columns `breast` and `breast-quad`
subview2 = SubViewClassifier(DecisionTreeClassifier(random_state=10), ["age", "menopause", "irradiant"], mode='columns') # The columns `age`, `menopause`, and `irradiant`
subview3 = SubViewClassifier(DecisionTreeClassifier(random_state=10), ".*", mode='regex') # All the columns
# Create a TriTraining wrapper with the subviews
tritraining = TriTraining(base_estimator=[subview1, subview2, subview3], random_state=10)
```

The wrappers that support multi-learning can be used with the subview adapters. The adapters are in the `sslearn.subview` module. 


### Comparison example

A comparison example is available in the [Jupyter Notebook](https://colab.research.google.com/drive/1wKSz-f7N4elqQwz_phrWXDrf3lRqaD6s#sandboxMode=true&scrollTo=L4vJsnE0AwVE). Here is a snippet of the code with the dataset already loaded.

The propose of this code is compare the accuracy of the algorithms in a semi-supervised dataset. It is used a 10-fold cross-validation to get the accuracy of each algorithm, and with the 10%, 20%, 30%, and 40% of the instances labeled.


```python
# The dataset is already loaded, the models are built and the results objects have been created.

# First statified k-fold
skf = StratifiedKFold(n_splits=10)
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    for i, lr in enumerate([0.1, 0.2, 0.3, 0.4]):
      # Create the semi-supervised dataset
      X_ss, y_ss, _, _ = artificial_ssl_dataset(X_train, y_train, lr, 1)
      for name, model in models.items():
        # Fit the model
        model.fit(X_ss, y_ss)
        # Score the results and save it
        results[name][i].append(model.score(X_test, y_test))
```

The average results are:

|                          | 10%    | 20%    | 30%    | 40%    |
|--------------------------|---------|---------|---------|---------|
| Self-Training            | 89.99% | 90.52% | 91.05% | 89.81% |
| Setred                   | 88.76% | 90.86% | 90.86% | 91.04% |
| Co-Training              | 90.69% | 92.27% | 91.75% | 91.75% |
| Co-Training by Committee | 91.22% | 92.62% | 91.75% | 91.75% |
| Democratic Co-Learning   | 91.92% | 92.80% | 93.68% | 94.20% |
| RASCO                    | 90.86% | 91.74% | 94.38% | 93.33% |
| RelRASCO                 | 90.87% | 93.15% | 92.45% | 93.50% |
| CoForest                 | 91.39% | 92.80% | 92.63% | 92.45% |
| TriTraining              | 91.56% | 90.68% | 91.04% | 91.75% |
| DeTriTraining            | 85.24% | 85.06% | 85.06% | 85.24% |

## Restricted set classification

The RSC algorithms support datasets with pairwise constraints. The library provides the `WhoIsWhoClassifier`, the `feature_fusion` and `probability_fusion` methods.
`WhoIsWhoClassifier` is a wrapper that uses the RSC algorithms and supports only cannot-link constraints. The `feature_fusion` and `probability_fusion` methods supports both constraints but only in prediction time. All algorithms are in the `sslearn.restriced` module.

The complete example is avaliable in the [Jupyter Notebook](https://colab.research.google.com/drive/1wKSz-f7N4elqQwz_phrWXDrf3lRqaD6s#sandboxMode=true&scrollTo=sN3XUw4K_it-). Here is a snippet of the code with the dataset and the constraints already loaded.

```python
wiw = WhoIsWhoClassifier(base_estimator=DecisionTreeClassifier(random_state=10),
                         method="hungarian")  # The method can ben `hungarian` and `greedy`
wiw.fit(X_train, y_train, cannot_link_train)
y_predicted = wiw.predict(X_test, cannot_link_test)
```
For `WhoIsWhoClassifier` the cannot-link constraints must be an array that the size is the number of instances in the dataset. Each index is mapped to the instance in the dataset. If two instances or more cannot-link the value must be the same in the array.

For the `feature_fusion` and `probability_fusion` methods, the constraints must be in dictionary format. The values are the instances that cannot-link or must-link. The keys represent the group of instances, but normaly is the first instance in the group (to ensure differents groups not have the same key). That methods requires a pre-trained model.

```python
estimator = DecisionTreeClassifier(random_state=10)
estimator.fit(X_train, y_train)
y_predicted_f = feature_fusion(estimator, X_test, must_link_test_dict, cannot_link_test_dict)
y_predicted_p = probability_fusion(estimator, X_test, must_link_test_dict, cannot_link_test_dict)
```

`feature_fusion` combine the features of the instances that must-link replacing the features for the average of the features in the group. With `probability_fusion` the post-probabilities of the instances that must-link are averaged. In both cases it uses the hungarian algorithm to solve the cannot-link constraints.