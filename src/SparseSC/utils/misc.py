# Allow capturing output
# Modified (to not capture stderr too) from https://stackoverflow.com/questions/5136611/
import contextlib
import sys

from .print_progress import it_progressbar, it_progressmsg

@contextlib.contextmanager
def capture():
    STDOUT = sys.stdout
    try:
        sys.stdout = DummyFile()
        yield 
    finally:
        sys.stdout = STDOUT


class DummyFile(object):
    def write(self, x):
        pass


@contextlib.contextmanager
def capture_all():
    STDOUT, STDERR = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = DummyFile(), DummyFile()
        yield 
    finally:
        sys.stdout, sys.stderr = STDOUT, STDERR


def par_map(part_fn, it, F, loop_verbose, n_multi=0, header="LOOP"):  
    if n_multi>0:
        from multiprocessing import Pool

        with Pool(n_multi) as p:
            #p.map evals the it so can't use it_progressbar(it)
            if loop_verbose==1:
                rets = []
                print(header + ":")
                for ret in it_progressbar(p.imap(part_fn, it), count=F):
                    rets.append(ret)
            elif loop_verbose==2:
                rets = []
                for ret in it_progressmsg(p.imap(part_fn, it), prefix=header, count=F):
                    rets.append(ret)
            else:
                rets = p.map(part_fn, it)
    else:
        if loop_verbose==1:
            print(header + ":")
            it = it_progressbar(it, count=F)
        elif loop_verbose==2:
            it = it_progressmsg(it, prefix=header, count=F)
        rets = list(map(part_fn, it))
    return rets


class PreDemeanScaler:
    """
    Units are defined by rows and cols are "pre" and "post" separated.
    Demeans each row by the "pre" mean.
    """

    # maybe fit should just take Y and T0 (in init())?
    # Try in sklearn.pipeline with fit() for that and predict (on default Y_post)
    # might want wrappers around fit to make that work fine with pipeline (given its standard arguments).
    # maybe call the vars X rather than Y?
    def __init__(self):
        self.means = None
        # self.T0 = T0

    def fit(self, Y):
        """
        Ex. fit(Y.iloc[:,0:T0])
        """
        import numpy as np

        self.means = np.mean(Y, axis=1)

    def transform(self, Y):
        return (Y.T - self.means).T

    def inverse_transform(self, Y):
        return (Y.T + self.means).T


def _ensure_good_donor_pool(custom_donor_pool, control_units):
    N0 = custom_donor_pool.shape[1]
    custom_donor_pool_c = custom_donor_pool[control_units, :]
    for i in range(N0):
        custom_donor_pool_c[i, i] = False
    custom_donor_pool[control_units, :] = custom_donor_pool_c
    return custom_donor_pool


def _get_fit_units(model_type, control_units, treated_units, N):
    if model_type == "retrospective":
        return control_units
    elif model_type == "prospective":
        return range(N)
    elif model_type == "prospective-restricted:":
        return treated_units
    # model_type=="full":
    return range(N)  # same as control_units

