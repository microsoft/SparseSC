"""Store the typical information for a match.
"""
from collections import namedtuple

from SparseSC.utils.dist_summary import SSC_DescrStat

MatchingEstimate = namedtuple(
    'MatchingEstimate', 'att_est att_debiased_est atut_est atut_debiased_est ate_est ate_debiased_est aa_est naive_est')
#"""
#aa = The difference between control counterfactuals for controls and controls (Y_c_cf_c  - Y_c)
#Debiased means we subtract from the estimate the aa estimate
#att = Average Treatment effect on the Treated (Y_t - Y_t_cf_c).
#atut = Average Treatment on the UnTreated (Y_c_cf_t - Y_c)
#ate = Average Treatment Effect (pooling att and atut samples)
#naive = Just comparing Y_t and Y_c (no matching). Helpful for comparison (gauging selection size)
#"""

class DescrSet:
    """Holds potential distribution summaries for the various data used for matching
    """

    def __init__(self, descr_Y_t=None, descr_Y_t_cf_c=None, descr_Y_diff_t_cf_c=None, 
                 descr_Y_c=None, descr_Y_diff_t_c=None, descr_Y_c_cf_c=None, descr_Y_diff_c_cf_c=None, 
                 descr_Y_c_cf_t=None, descr_Y_diff_c_cf_t=None):
        """Generate the common descriptive stats from data and differences
        :param descr_Y_t: SSC_DescrStat for Y_t (outcomes for treated units)
        :param descr_Y_t_cf_c: SSC_DescrStat for Y_t_cf_c (outcomes for (control) counterfactuals of treated units)
        :param descr_Y_diff_t_cf_c: SSC_DescrStat of Y_t - Y_t_cf_c
        :param descr_Y_c: SSC_DescrStat of Y_c (outcomes for control units)
        :param descr_Y_diff_t_c: SSC_DescrStat for Y_t-Y_c
        :param descr_Y_c_cf_c: SSC_DescrStat for Y_c_cf_c (outcomes for (control) counterfactuals of control units)
        :param descr_Y_diff_c_cf_c: SSC_DescrStat for Y_c - Y_c_cf_c
        :param descr_Y_c_cf_t: SSC_DescrStat for Y_c_cf_t (outcomes for (TREATED) counterfactuals of control units)
        :param descr_Y_diff_c_cf_t: SSC_DescrStat for Y_c_cf_t - Y_c
        """
        self.descr_Y_t = descr_Y_t
        self.descr_Y_t_cf_c = descr_Y_t_cf_c
        self.descr_Y_diff_t_cf_c = descr_Y_diff_t_cf_c

        self.descr_Y_c = descr_Y_c
        self.descr_Y_diff_t_c = descr_Y_diff_t_c
        self.descr_Y_c_cf_c = descr_Y_c_cf_c
        self.descr_Y_diff_c_cf_c = descr_Y_diff_c_cf_c

        self.descr_Y_c_cf_t = descr_Y_c_cf_t
        self.descr_Y_diff_c_cf_t = descr_Y_diff_c_cf_t

    def __repr__(self):
        return ("%s(descr_Y_t=%s, descr_Y_t_cf_c=%s, descr_Y_diff_t_cf_c=%s, descr_Y_c=%s" +
                "descr_Y_diff_t_c=%s, descr_Y_c_cf_c=%s, descr_Y_diff_c_cf_c=%s, descr_Y_c_cf_t=%s," + 
                " descr_Y_diff_c_cf_t=%s)") % (self.__class__.__name__, self.descr_Y_t, 
                                               self.descr_Y_t_cf_c, self.descr_Y_diff_t_cf_c, self.descr_Y_c, 
                                               self.descr_Y_diff_t_c, self.descr_Y_c_cf_c, self.descr_Y_diff_c_cf_c, 
                                               self.descr_Y_c_cf_t, self.descr_Y_diff_c_cf_t)

    @staticmethod
    def from_data(Y_t=None, Y_t_cf_c=None, Y_c=None, Y_c_cf_c=None, Y_c_cf_t=None):
        """Generate the common descriptive stats from data and differences
        :param Y_t: np.array of dim=(N_t, T) for outcomes for treated units
        :param Y_t_cf_c: np.array of dim=(N_t, T) for outcomes for (control) counterfactuals of treated units (used to get the average treatment effect on the treated (ATT))
        :param Y_c: np.array of dim=(N_c, T) for outcomes for control units
        :param Y_c_cf_c: np.array of dim=(N_c, T) for outcomes for (control) counterfactuals of control units (used for AA test)
        :param Y_c_cf_t: np.array of dim=(N_c, T) for outcomes for (TREATED) counterfactuals of control units (used to calculate average treatment effect on the untreated (ATUT), or pooled with ATT to get the average treatment effect (ATE))
        :returns: DescrSet
        """
        # Note: While possible, there's no real use for treated matched to other treateds.
        def _gen_if_valid(Y):
            return SSC_DescrStat.from_data(Y) if Y is not None else None

        def _gen_diff_if_valid(Y1, Y2):
            return SSC_DescrStat.from_data(Y1-Y2) if (Y1 is not None and Y2 is not None) else None

        descr_Y_t = _gen_if_valid(Y_t)
        descr_Y_t_cf_c = _gen_if_valid(Y_t_cf_c)
        descr_Y_diff_t_cf_c = _gen_diff_if_valid(Y_t, Y_t_cf_c)

        descr_Y_c = _gen_if_valid(Y_c)
        descr_Y_diff_t_c = _gen_diff_if_valid(Y_t, Y_c)
        descr_Y_c_cf_c = _gen_if_valid(Y_c_cf_c)
        descr_Y_diff_c_cf_c = _gen_diff_if_valid(Y_c_cf_c, Y_c)
        descr_Y_c_cf_t = _gen_if_valid(Y_c_cf_t)
        descr_Y_diff_c_cf_t = _gen_diff_if_valid(Y_c_cf_t, Y_c)

        return DescrSet(descr_Y_t, descr_Y_t_cf_c, descr_Y_diff_t_cf_c, 
                        descr_Y_c, descr_Y_diff_t_c, descr_Y_c_cf_c, descr_Y_diff_c_cf_c, 
                        descr_Y_c_cf_t, descr_Y_diff_c_cf_t)

    def __add__(self, other):
        def _add_if_valid(a, b):
            return a+b if (a is not None and b is not None) else None
        return DescrSet(descr_Y_t=_add_if_valid(self.descr_Y_t, other.descr_Y_t), 
                        descr_Y_t_cf_c=_add_if_valid(self.descr_Y_t_cf_c, other.descr_Y_t_cf_c), 
                        descr_Y_diff_t_cf_c=_add_if_valid(self.descr_Y_diff_t_cf_c, other.descr_Y_diff_t_cf_c), 
                        descr_Y_c=_add_if_valid(self.descr_Y_c, other.descr_Y_c), 
                        descr_Y_diff_t_c=_add_if_valid(self.descr_Y_diff_t_c, other.descr_Y_diff_t_c), 
                        descr_Y_c_cf_c=_add_if_valid(self.descr_Y_c_cf_c, other.descr_Y_c_cf_c), 
                        descr_Y_diff_c_cf_c=_add_if_valid(self.descr_Y_diff_c_cf_c, other.descr_Y_diff_c_cf_c), 
                        descr_Y_c_cf_t=_add_if_valid(self.descr_Y_c_cf_t, other.descr_Y_c_cf_t), 
                        descr_Y_diff_c_cf_t=_add_if_valid(self.descr_Y_diff_c_cf_t, other.descr_Y_diff_c_cf_t))

    def calc_estimates(self):
        """ Takes matrices of effects for multiple events and return averaged results
        """
        def _calc_estimate(descr_stat1, descr_stat2):
            if descr_stat1 is None or descr_stat2 is None:
                return None
            return SSC_DescrStat.lcl_comp_means(descr_stat1, descr_stat2)

        att_est = _calc_estimate(self.descr_Y_t, self.descr_Y_t_cf_c)
        att_debiased_est = _calc_estimate(self.descr_Y_diff_t_cf_c, self.descr_Y_diff_c_cf_c)

        atut_est = _calc_estimate(self.descr_Y_c_cf_t, self.descr_Y_c)
        atut_debiased_est = _calc_estimate(self.descr_Y_diff_c_cf_t, self.descr_Y_diff_c_cf_c)

        ate_est, ate_debiased_est = None, None
        if all(d is not None for d in [self.descr_Y_t, self.descr_Y_c_cf_t, self.descr_Y_t_cf_c, self.descr_Y_c]):
            ate_est = _calc_estimate(self.descr_Y_t + self.descr_Y_c_cf_t,
                                                   self.descr_Y_t_cf_c + self.descr_Y_c)
        if all(d is not None for d in [self.descr_Y_diff_t_cf_c, self.descr_Y_diff_c_cf_t, self.descr_Y_diff_c_cf_c]):    
            ate_debiased_est = _calc_estimate(self.descr_Y_diff_t_cf_c +
                                                   self.descr_Y_diff_c_cf_t, self.descr_Y_diff_c_cf_c)

        aa_est = _calc_estimate(self.descr_Y_c_cf_c, self.descr_Y_c)
        # descr_Y_diff_c_cf_c #used for the double comparisons

        naive_est = _calc_estimate(self.descr_Y_t, self.descr_Y_c)
        # descr_Y_diff_t_c #Don't think this could be useful, but just in case.

        return MatchingEstimate(att_est, att_debiased_est, 
                                atut_est, atut_debiased_est,
                                ate_est, ate_debiased_est,
                                aa_est, naive_est)
