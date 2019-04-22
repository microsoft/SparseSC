# Anomaly Detection

### Overview

In this scenario the goal is to identify irregular values in an outcome
variable prospectively in a homogeneous population (i.e. when no
treatment / intervention is planned).  As an example, we may wish to detect
failure of any one machine in a cluster, and to do so, we wish to create a
synthetic unit for each machine which is composed of a weighted average of
other machines in the cluster.  In particular, there may be variation of
the workload across the cluster and where workload may vary across the
cluster by (possibly unobserved) differences in machine hardware, cluster
architecture, scheduler versions, networking architecture, job type, etc. 

Like the Prospective Treatment Effects scenario, *Feature* data consist of
of unit attributes (covariates) and a subset of the pre-intervention values
from the outcome of interest, and **target** data consist of the remaining
pre-intervention values for the outcome of interest, and Cross fold
validation is conducted using the entire dataset, and Cross validation and
gradient folds are determined randomly. 

### Example

In this scenario, we'll need a matrix with past observations of the outcome
(target) of interest (`targets`), with one row per unit of observation, and
one column per time period, ordered from left to right.  Additionally we
may have another matrix of additional features with one row per unit and
one column per feature (`features`).  Armed with this we may wish to construct a
synthetic control model to help decide weather future observations
(`additional_observations`) deviate from their synthetic predictions.

The strategy will be to divide the `targets` matrix into two parts (before
and after column `t`), one of which will be used as features, and other
which will be treated as outcomes for the purpose of fitting the weights
which make up the synthetic controls model.

```python
from numpy import hstack
from SparseSC import fit

# Let X be the features plus some of the targets
X = hstack([features, targets[:,:t])

# And let Y be the remaining targets
Y = targets[:,t:]

# fit the model:
fitted_model = fit(X=X,
                   Y=Y,
                   model_type="full")
```

The `model_type="full"` allows produces a model in which every unit can
serve as a control for every other unit, unless of course the parameter
`custom_donor_pool` is specified.

Now with our fitted synthetic control model, as soon as new set of targets
outcomes are observed for each unit, we can create synthetic outcomes using
our fitted model using the `predict()` method:

```python
synthetic_controls = fitted_model.predict(additional_observations)
```

Note that while the call to `fit()` is computationally intensive, the call
to `model.predict()` is fast and can be used for real time anomaly
detection.

### Model Details:

This model yields a synthetic unit for every unit in the dataset, and
synthetic units are composted of the remaining units not included in the
same gradient fold.

| Type | Units used to fit V & penalties | Donor pool for W |
|---|---|---|
|(prospective) full|All units|All units|
