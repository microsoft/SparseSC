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
