import unittest
import numpy as np

try:
    import SparseSC
except ImportError:
    raise RuntimeError("SparseSC is not installed. Use 'pip install -e .' or 'conda develop .' from repo root to install in dev mode")


class TestNormalForErrors(unittest.TestCase):
    def test_SSC_DescrStat(self):
        mat = np.arange(20).reshape(4,5)
        ds_top = SparseSC.SSC_DescrStat.from_data(mat[:3,:])
        ds_bottom = SparseSC.SSC_DescrStat.from_data(mat[3:,:])
        ds_add = ds_top + ds_bottom
        ds_top.update(mat[3:,:])
        ds_whole = SparseSC.SSC_DescrStat.from_data(mat)
        assert ds_add == ds_top, "ds_add == ds_top"
        assert ds_add == ds_whole, "ds_add == ds_whole"

    def test_DescrSet(self):
        Y_t = (np.arange(20)+1).reshape(4,5)
        Y_c = np.arange(20).reshape(4,5)
        Y_t_cf_c = Y_c
        ds = SparseSC.DescrSet.from_data(Y_t=Y_t, Y_t_cf_c=Y_t_cf_c)
        est = ds.calc_estimates()
        assert np.array_equal(est.att_est.effect, np.full((5), 1.0)), "estimations not right"

if __name__ == "__main__":
    t = TestNormalForErrors()
    t.test_SSC_DescrStat()
    t.test_DescrSet()

    #unittest.main()
