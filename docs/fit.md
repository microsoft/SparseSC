
# Overview of Fit.py

## The `model_type` parameter 

There are three distinct types of model fitting with respect choosing
optimal values of the penalty parameters and the collection of units
that are used for cross validation. 

Recall that a synthetic treatment unit is defined as a weighted average of
control units where weights are determined from the targets and are chosen
to minimize the difference between the each the treated unit and its
synthetic unit in the absence of an intervention.  This methodology can be
combined with cross-fold validation a variety of ways 
separate use cases.

### Retrospective Treatment Effects:  ( *model_type = "retrospective"*)

In a retrospective analysis, a subset of units have received a treatment
which possibly correlates with features of the units and values for the
target variable have been collected both before and after an intervention.
For example, a software update may have been applied to machines in a
cluster which were experiences unusually latency, and retrospectively an
analyst wishes to understand the effect of the update on another outcome
such as memory utilization. 

In this scenario, for each treated unit we wish to create a synthetic unit
composed only of untreated units.  The units are divided into a training set
consisting of just the control units, and a test set consisting of the
treated units.  Within the training set, *feature* data will consist of of
unit attributes (covariates) and pre-intervention values from the outcome
of interest. Likewise, **target** data consist of post-intervention values
for the outcome of interest.

Cross-fold validation is done within the training set, holding out a single
fold, identifying feature weights within the remaining folds, creating
synthetic units for each held out unit defined as a weighted average of the
non-held-out units. Out-of-sample prediction errors are calculated for each
training fold and summed across folds. The set of penalty parameters that
minimizes the Cross-Fold out-of-sample error are chosen.  Feature weights
are calculated within the training set for the chosen penalties, and
finally individual synthetic units are calculated for each treated unit
which is a weighted average of the control units.

This model yields a synthetic unit for each treated unit composed of
control units. 

### Prospective Treatment Effects ( *model_type = "prospective"*)

In a prospective analysis, a subset of units have been designated to
receive a treatment but the treatment has not yet occurred and the
designation of the treatment may be correlated with a (possibly unobserved)
feature of the treatment units.  For example, a software update may have
been planned for machines in a cluster which are experiencing unusually
latency, and there is a desire to understand the impact of the software on
memory utilization.

Like the retrospective scenario, for each treated unit we wish to create a
synthetic unit composed only of untreated units.

*Feature* data consist of of unit attributes (covariates) and a subset
of the pre-intervention values from the outcome of interest, and **target**
data consist of the remaining pre-intervention values for the outcome of
interest

Cross fold validation is conducted using the entire dataset without regard
to intent to treat.  However, treated units allocated to a single gradient
fold, ensuring that synthetic treated units are composed of only the
control units.

Cross-fold validation is done within the *test* set, holding out a single
fold, identifying feature weights within the remaining treatment folds
combined with the control units, synthetic units for each held out unit
defined as a weighted average of the full set of control units. 

Out-of-sample prediction errors are calculated for each treatment fold and
the sum of these defines the Cross-Fold out-of-sample error. The set of
penalty parameters that minimizes the Cross-Fold out-of-sample error are
chosen.  Feature weights are calculated within the training set for the
chosen penalties, and finally individual synthetic units are calculated for
each full unit which is a weighted average of the control units.

This model yields a synthetic unit for each treated unit composed of
control units. 

### Prospective Treatment Effects training (*model_type = "prospective-restricted"*)

This is motivated by the same example as the previous sample.  It requires
a larger set of treated units for similar levels of precision, with the
benefit of substantially faster running time.

The units are divided into a training set consisting of just the control
units, and a test set consisting of the unit which will be treated.
*feature* data will consist of of unit attributes (covariates) and a subset
of the pre-intervention values from the outcome of interest, and **target**
data consist of the remaining pre-intervention values for the outcome of
interest

Cross-fold validation is done within the *test* set, holding out a single
fold, identifying feature weights within the remaining treatment folds
combined with the control units, synthetic units for each held out unit
defined as a weighted average of the full set of control units. 

Out-of-sample prediction errors are calculated for each treatment fold and
the sum of these defines the Cross-Fold out-of-sample error. The set of
penalty parameters that minimizes the Cross-Fold out-of-sample error are
chosen.  Feature weights are calculated within the training set for the
chosen penalties, and finally individual synthetic units are calculated for
each full unit which is a weighted average of the control units.

This model yields a synthetic unit for each treated unit composed of
control units. 

Not that this model will tend to have wider confidence intervals and small estimated treatments given the sample it is fit on.

### Prospective Failure Detection ( *model_type = "full"*)

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

This model yields a synthetic unit for every unit in the dataset, and
synthetic units are composted of the remaining units not included in the
same gradient fold. 

### Summary

Here is a summary of the main differences between the model types.

| Type | Sample fitting (V & penalties) | Donor pool for W |
|---|---|---|
|retrospective|Controls|Controls|
|prospective|All|Controls|
|prospective-restricted|Treated|Controls|
|(prospective) full|All|All|

A tree view of differences:
* Treatment date: The *prospective* studies differ from the *retrospective* study in that they can use all units for fitting.
* (Prospective studies) Treated units: The intended-to-treat (*ITT*) studies differ from the *full* in that the "treated" units can't be used for donors.
* (Prospective-ITT studies): The *restrictive* model differs in that it tries to maximize predictive power for just the treated units.
