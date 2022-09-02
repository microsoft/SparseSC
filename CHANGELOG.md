# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- Separate headings for Added/Changed/Removed/Fixed/Deprecated/Security -->
## [Unreleased](https://github.com/Microsoft/SparseSC/compare/v0.2.0...master)
### Added
- Added `fit_args` to `estimate_effect()` that can be used with `fit_fast()` with the default variable weight algorithm (`sklearn`'s [MTLassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html)). Common usages would include increasing `max_iter` (if you want to improve convergence) or `n_jobs` (if you want to run in parallel).
- Separated CV folds from Cross-fitting folds in `estimate_effects()`. `cv_folds` controls the amount of additional estimation done on the control units (this used to be controled by `max_n_pl`, but that parameter now only governs the amount of post-estimation processing that is done). This will do `cv_folds` extra estimations per treatment time period, though if `=1` then no extra work will be done (but control residuals might be biased toward 0).
- Added additional option `Y_col_block_size` to `MTLassoCV_MatchSpace_factory` to estimate `V` on block-averages of `Y` (e.g. taking a 150 cols down to 5 by doing averages over 30 cols at a time).
- Added `se_factor` to `MTLassoCV_MatchSpace_factory` to use a different penalty than the MSE min.
- For large data, approximate the outcomes using a normal distribution (`DescrSet`), and allow for calculating estimates. 

## 0.2.0 - 2020-05-06
### Added
- Added tools too to use `fit_fast()` with large datasets. This includes the `sample_frac` option to  `MTLassoCV_MatchSpace_factory()` to estimate the match space on a subset of the observations. It also includes `fit_fast()` option `avoid_NxN_mats` which will avoid making large matrices (at the expense of only returning the Synthetic control `targets` and `targets_aux` and not the full weight matrix)
- Added logging in `fit_fast` via the `verbose` numerical option. This can help identify out-of-memory errors.
- Added a pseudo-Doubly robust match space maker `D_LassoCV_MatchSpace_factory`. It apporptions some of the normalized variable V weight to those variables that are good predictors of treatment. This should only be done if there are many treated units so that one can reasonably model this relationship. 
- Switched using standardized Azure Batch config library


## 0.1.0 - 2019-07-25
Initial release.
