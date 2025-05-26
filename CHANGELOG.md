# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0.0] - 2025-05-26

### Added
- Included `get_dataset_regresion` for separation of labeled and unlabeled data. For the regresion dataset the unlabeled instances is marked with `NaN`; by @aliciaolivaresgil.
- Includes `check_regressor` in the `sslearn.utils` module to check if the estimator is a regressor or not; by @aliciaolivaresgil.
- The regressor `CoReg` is now included, described in the paper "Semi-supervised regression with co-training" by Zhi-Hua Zhou and Ming Li. Implemented in the `sslearn.wrapper` module; by @aliciaolivaresgil.
- The regresion `TriTrainingRegressor` is now included, made by adapting the `TriTraining` method to regression tasks created by Alicia Olivares Gil. Implemented in the `sslearn.wrapper` module; by @aliciaolivaresgil.

### Changed
- The functions `read_keel` and `read_csv` from the `sslearn.datasets` module now has a new parameter `is_regression` to indicate if the dataset is for regression or classification. This will change the way the dataset is processed and returned; by @aliciaolivaresgil.

### Fixed

- Repaired `restricted.feature_fusion` and `restricted.probability_fusion` when the instances that cannot link are not consecutives.

### Testing 
- Added tests for the new regression methods and the changes in the dataset reading functions; by @aliciaolivaresgil.

## [1.0.5.2] - 2024-05-27

### HotFix

- Remove some files that are not necessary in the package.

## [1.0.5.1] - 2024-05-20

### Fixed

- Fixed bugs in `artificial_ssl_dataset`, now support again pandas DataFrame and y_unlabeled returns the right values

## [1.0.5] - 2024-05-08

### Added
- `feature_fusion` and `probability_fusion` methods for restricted in `sslearn.restricted` module. 

### Fixed
- CoForest random integer is now compatible with Windows.

## [1.0.4] - 2024-01-31

### Added
- Add a parameter to `artificial_ssl_dataset` to force a minimum of instances. Issue #11
- Add a parameter to `artificial_ssl_dataset` to return indexes. Issue #13

### Changed
- The `artificial_ssl_dataset` changed the process to generate the dataset, based in indexes. Issue #13

### Fixed
- DeTriTraining now is vectorized and is faster than before.

## [1.0.3.1] - 2023-04-01

### Changed
- Hot fix for avoid problems with Pypi

## [1.0.3] - 2023-03-29

### Added
- Methods now support no unlabeled data. In this case, the method will return the same as the base estimator.

### Changed
- In OneHotEncoder, the `sparse` parameter is now `sparse_output` to avoid a FutureWarning.

### Fixed

- CoForest now is most similar to the original paper.
- TriTraining can use at least 3 n_jobs. Fixed the bug that allows using as many n_jobs as cpus in the machine.

## [1.0.2] - 2023-02-17

### Fixed

- Fixed a bug in TriTraining when one of the base estimators has not a random_state parameter.
- Fixed OneVsRestSSL with the random_state parameter.
- Fixed WiWTriTraining when no `instance_group` parameter is not provided.
- Fixed a FutureWarning for `sparse` parameter in `OneHotEncoder`. Changed to `sparse_output`.

## [1.0.1] - 2023-02-10

### Added

- CoTraining support a `threshold` parameter (default to 0.5) to control the threshold for adding new instances in the next iteration.

### Fixed

- Fixed a bug in CoTraining using LabelEncoder.