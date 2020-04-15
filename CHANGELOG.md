# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!--## [Unreleased] >
<!-- Separate headings for Added/Changed/Removed/Fixed/Deprecated/Security -->

## 0.2.0 - 2020-04-09
### Added
- Added tools too to use `fit_fast()` with large datasets. This includes `sample_frac` options to  `MTLassoCV_MatchSpace_factory()` to estimate the match space on a subset of the data and the `fit_fast()` option `avoid_NxN_mats` which will avoid making large matrices (at the expense of only returning the Synthetic control outcomes and not the full weight matrix)
- Added logging in `fit_fast` via the `verbose` numerical option.
- Added a pseudo-Doubly robust match space maker `D_LassoCV_MatchSpace_factory`. It apporptions some of the normalized variable V weight to those variables that are good predictors of treatment. This should only be done if there are many treated units so that one can reasonably model this relationship. 


## 0.1.0 - 2019-07-25
Initial release.

<!--[Unreleased]: Get compare link for Github. For VSTS it's https://econpricingengine.visualstudio.com/_git/PricingEngine/branches?baseVersion=GTv2.1.0&targetVersion=GBmaster&_a=commits >
