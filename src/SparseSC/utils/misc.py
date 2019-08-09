#Allow capturing output
#Modified (to not capture stderr too) from https://stackoverflow.com/questions/5136611/
import contextlib
@contextlib.contextmanager
def capture():
    import sys
    oldout = sys.stdout
    try:
        if sys.version_info[0] < 3:
            from cStringIO import StringIO
        else:
            from io import StringIO
        out=StringIO()
        sys.stdout = out
        yield out
    finally:
        sys.stdout = oldout
        out = out.getvalue()

@contextlib.contextmanager
def capture_all():
    import sys
    oldout,olderr = sys.stdout, sys.stderr
    try:
        if sys.version_info[0] < 3:
            from cStringIO import StringIO
        else:
            from io import StringIO
        out=[StringIO(), StringIO()]
        sys.stdout,sys.stderr = out
        yield out
    finally:
        sys.stdout,sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()

class PreDemeanScaler:
    '''
    Units are defined by rows and cols are "pre" and "post" separated.
    Demeans each row by the "pre" mean.
    '''
    #maybe fit should just take Y and T0 (in init())? 
    # Try in sklearn.pipeline with fit() for that and predict (on default Y_post)
    # might want wrappers around fit to make that work fine with pipeline (given its standard arguments).
    #maybe call the vars X rather than Y?
    def __init__(self):
        self.means = None
        #self.T0 = T0

    def fit(self, Y):
        '''
        Ex. fit(Y.iloc[:,0:T0])
        '''
        import numpy as np
        self.means = np.mean(Y, axis=1)
    
    def transform(self, Y):
        return (Y.T - self.means).T

    def inverse_transform(self, Y):
        return (Y.T + self.means).T
