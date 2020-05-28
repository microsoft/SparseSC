"""This is a way to summarize (using normal approximations) the distributions of real and 
synthetic controls so that all data doesn't have to be stored.
"""
import math
from collections import namedtuple

import statsmodels
import numpy as np

# TODO: 
# - Allow passed in vector and return scalar

def tstat_generic(mean1, mean2, stdm, dof):
    """Vectorized version of statsmodels' _tstat_generic
    :param mean1: int or np.array
    :param mean2: int or np.array
    :param stdm: int or np.array of the standard deviation of the pooled sample
    :param dof: int
    """
    from statsmodels.stats.weightstats import _tstat_generic
    l = len(mean1)
    if l == 1:
        tstat, pval = _tstat_generic(mean1, mean2, stdm, dof, 'two-sided', diff=0)
    else:
        tstat, pval = zip(*[_tstat_generic(mean1[i], mean2[i], stdm[i], dof, 'two-sided', diff=0) 
                            for i in range(l)])
    #tstat = (mean1 - mean2) / stdm #
    #pvalue = stats.t.sf(np.abs(tstat), dof)*2
    # cohen's d: diff/samplt std dev
    return tstat, pval

# def pooled_variances_scalar(sample_variances, sample_sizes):
#  """Estimate pooled variance from a set of samples. Assumes same variance but allows different means."""
#  return np.average(sample_variances, weights=(sample_sizes-1))

def pooled_variances(sample_variances, sample_sizes):
    """Estimate pooled variance from a set of samples. Assumes same variance but allows different means.
    If inputs are nxl then return l
    """
    # https://en.wikipedia.org/wiki/Pooled_variance
    return np.average(sample_variances, weights=(sample_sizes-1), axis=0)

Estimate = namedtuple('Estimate', 'effect pval') # can hold scalars or vectors

class SSC_DescrStat(object):
    """Stores mean and variance for a sample in a way that can be updated
    with new observations or adding together summaries. Similar to statsmodel's DescrStatW
    except we don't keep the raw data, and we use 'online' algorithm's to allow for incremental approach."""
    # Similar to https://github.com/grantjenks/python-runstats but uses numpy and doesn't do higher order stats and has multiple columns
    # Ref: https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.DescrStatsW.html#statsmodels.stats.weightstats.DescrStatsW

    def __init__(self, nobs, mean, M2):
        """
        :param nobs: scalar
        :param mean: vector
        :param M2: vector of same length as mean. Sum of squared deviations (sum_i (x_i-mean)^2). 
            Sometimes called 'S' (capital; though 's' is often sample variance)
        :raises: ValueError
        """
        import numbers
        if not isinstance(nobs, numbers.Number):
            raise ValueError('mean should be np vector')
        self.nobs = nobs
        if len(mean.shape)!=1:
            raise ValueError('mean should be np vector')
        if len(M2.shape)!=1:
            raise ValueError('M2 should be np vector')
        if len(M2.shape)!=len(mean.shape):
            raise ValueError('M2 and mean should be the same length')
        self.mean = mean
        self.M2 = M2  # sometimes called S (though s is often sample variance)

    def __repr__(self):
        return "%s(nobs=%s, mean=%s, M2=%s)" % (self.__class__.__name__, self.nobs, self.mean, self.M2)

    @staticmethod
    def from_data(data):
        """
        :param data: 2D np.array. We compute stats per column.
        :returns: SSC_DescrStat object
        """
        N = data.shape[0]
        mean = np.average(data, axis=0)
        M2 = np.var(data, axis=0)*N
        return SSC_DescrStat(N, mean, M2)

    def __add__(self, other, alt=False):
        """
        Chan's parallel algorithm
        :param other: Other SSC_DescrStat object
        :param alt: Use when roughly similar sizes and both are large. Avoid catastrophic cancellation 
        :returns: new SSC_DescrStat object
        """
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # TODO: could make alt auto (look at, e.g., abs(self.nobs - other.nobs)/new_n)
        new_n = self.nobs + other.nobs
        delta = other.mean - self.mean
        if not alt:
            new_mean = self.mean+delta*(other.nobs/new_n)
        else:
            new_mean = (self.nobs*self.mean + other.nobs*other.mean)/new_n 
        new_M2 = self.M2 + other.M2 + np.square(delta) * self.nobs * other.nobs / new_n
        return SSC_DescrStat(new_n, new_mean, new_M2)

    def update(self, obs):
        """Welford's online algorithm
        :param obs: new observation vector
        """
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.nobs += 1
        delta = obs - self.mean
        self.mean += delta*1./self.nobs
        self.M2 += delta*(obs - self.mean)

    def variance(self, ddof=1):
        """
        :param ddof: delta degree of difference. 1 will give sample variance (s).
        :returns: Variance
        """
        if self.nobs == 1:
            return np.zeros(self.mean.shape)
        return self.M2/(self.nobs-ddof)

    @property
    def var(self):
        return self.variance()

    def stddev(self, ddof=1):
        """Standard Deviation
        :param ddof: delta degree of difference. 1 will give sample Standard Deviation
        :returns: Standard Deviation
        """
        return np.sqrt(self.variance(ddof))
    
    @property
    def std(self):
        return self.stddev()

    def std_mean(self, ddof=1):
        """Standard Deviation/Error of the mean
        :param ddof: delta degree of difference. 1 will give sample variance (s).
        :returns: Standard Error
        """
        return self.stddev(ddof)/math.sqrt(self.nobs-1)
        
    @property
    def sumsquares(self):
        return self.M2

    @property
    def sum(self):
        return self.mean*self.nobs

    @property
    def sum_weights(self):
        return self.sum

    
    @staticmethod
    def lcl_comp_means(descr1, descr2):
        """ Calclulates the t-test  of the difference in means. Local version of statsmodels.stats.weightstats import CompareMeans
        :param descr1: DescrStatW-type object of sample statistics
        :param descr2: DescrStatW-type object of sample statistics
        """
        #from statsmodels.stats.weightstats import CompareMeans
        # Do statsmodels.CompareMeans with just summary stats. 
        var_pooled = pooled_variances(np.array([descr1.var, descr2.var]), np.array([descr1.nobs, descr2.nobs]))
        stdm = np.sqrt(var_pooled * (1. / descr1.nobs + 1. / descr2.nobs)) # ~samplt std dev/sqrt(N)
        dof = descr1.nobs - 1 + descr2.nobs - 1
        tstat, pval = tstat_generic(descr1.mean, descr2.mean, stdm, dof)
        effect = descr1.mean - descr2.mean
        return Estimate(effect, pval)
