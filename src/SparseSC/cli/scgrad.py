#!/usr/bin/env python
"""
A service like daemon for calculating components of the gradient
"""
# pylint: disable=invalid-name, unused-import, multiple-imports
import platform
import numpy as np
import uuid
import sys, time, os, atexit, json
from SparseSC.cli.daemon import Daemon
from yaml import load, dump
import tempfile


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# pluck = lambda d, *args: (d[arg] for arg in args)
def pluck(d, *args):
    """
    pluckr
    """
    # print("hello pluckr"); sys.stdout.flush()
    out = [None] * len(args)
    for i, key in enumerate(args):
        # print("key: " + key); sys.stdout.flush()
        try:
            out[i] = d[key]
        except KeyError:
            raise RuntimeError("no such key '{}'".format(key))
    return out


def grad_part(common, part, k):
    """
    Calculate a single component of the gradient
    """

    N0, N1, in_controls, splits, w_pen, treated_units, Y_treated, Y_control, dA_dV_ki, dB_dV_ki = pluck(
        common,
        "N0",
        "N1",
        "in_controls",
        "splits",
        "w_pen",
        "treated_units",
        "Y_treated",
        "Y_control",
        "dA_dV_ki",
        "dB_dV_ki",
    )

    in_controls2 = [np.ix_(i, i) for i in in_controls]

    A, weights, b_i = pluck(part, "A", "weights", "b_i")
    dA_dV_ki_k, dB_dV_ki_k = dA_dV_ki[k], dB_dV_ki[k]

    dPI_dV = np.zeros((N0, N1))  # stupid notation: PI = W.T

    try:
        for i, (_, (_, test)) in enumerate(zip(in_controls, splits)):

            dA = dA_dV_ki_k[i]
            dB = dB_dV_ki_k[i]
            try:
                b = np.linalg.solve(A[in_controls2[i]], dB - dA.dot(b_i[i]))
            except np.linalg.LinAlgError as exc:
                print("Unique weights not possible.")
                if w_pen == 0:
                    print("Try specifying a very small w_pen rather than 0.")
                raise exc
            dPI_dV[np.ix_(in_controls[i], treated_units[test])] = b
        # einsum is faster than the equivalent (Ey * Y_control.T.dot(dPI_dV).T.getA()).sum()
        return 2 * np.einsum(
            "ij,kj,ki->", (weights.T.dot(Y_control) - Y_treated), Y_control, dPI_dV
        )
    except Exception as err:
        print("{}: {}".format(err.__class__.__name__, getattr(err, "message", "<>")))
        raise RuntimeError("bye from scgrad")


DIR = "/tmp/" if platform.system() == "Darwin" else "/var/"
DAEMON_FIFO = "{}sc-daemon.fifo".format(DIR)
DAEMON_PID = "{}sc-gradient-daemon.pid".format(DIR)

_CONTAINER_OUTPUT_FILE = "output.yaml"  # Standard Output file
_GRAD_COMMON_FILE = "common.yaml"
_GRAD_PART_FILE = "part.yaml"

_BASENAMES = [_GRAD_COMMON_FILE, _GRAD_PART_FILE, _CONTAINER_OUTPUT_FILE]


class TestDaemon(Daemon):
    """
    A daemon which calculates Sparse SC gradient components
    """

    def run(self):
        print("run says hi: ")
        sys.stdout.flush()
        # pylint: disable=no-self-use
        while True:
            with open(DAEMON_FIFO, "r") as fifo:
                try:
                    params = fifo.read()
                    print("run says hi: " + params)
                    sys.stdout.flush()
                    tmpdirname, return_fifo, k = json.loads(params)
                    common_file, part_file, out_file = [
                        os.join(tmpdirname, name) for name in _BASENAMES
                    ]
                    print([common_file, part_file, out_file, return_fifo, k])
                    sys.stdout.flush()

                except:  # pylint: disable=bare-except

                    # SOMETHING WENT WRONG, RESPOND WITH A NON-ZERO
                    try:
                        with open(return_fifo, "w") as rf:
                            rf.write("1")
                    except:  # pylint: disable=bare-except
                        pass
                    else:
                        print("daemon something went wrong: ")
                        sys.stdout.flush()

                else:

                    # SEND THE SUCCESS RESPONSE
                    print("daemon all done: ")
                    sys.stdout.flush()
                    with open(return_fifo, "w") as rf:
                        rf.write("0")


class GradientDaemon(Daemon):
    """
    A daemon which calculates Sparse SC gradient components
    """

    def run(self):
        # pylint: disable=no-self-use
        while True:
            with open(DAEMON_FIFO, "r") as fifo:
                try:
                    params = fifo.read()
                    print("params: " + params)
                    sys.stdout.flush()
                    tmpdirname, return_fifo, k = json.loads(params)
                    print(_BASENAMES)
                    for file in os.listdir(tmpdirname):
                        print(file)
                    common_file, part_file, out_file = [
                        os.path.join(tmpdirname, name) for name in _BASENAMES
                    ]
                    print([common_file, part_file, out_file, return_fifo, k])
                    sys.stdout.flush()

                    # LOAD IN THE INPUT FILES
                    with open(common_file, "r") as fp:
                        common = load(fp, Loader=Loader)
                    with open(part_file, "r") as fp:
                        part = load(fp, Loader=Loader)

                    # DO THE WORK
                    print("about to do work: ")
                    sys.stdout.flush()
                    grad = grad_part(common, part, int(k))
                    print("did work: ")
                    sys.stdout.flush()

                    # DUMP THE RESULT TO THE OUTPUT FILE
                    with open(out_file, "w") as fp:
                        fp.write(dump(grad, Dumper=Dumper))

                except Exception as err:  # pylint: disable=bare-except

                    # SOMETHING WENT WRONG, RESPOND WITH A NON-ZERO
                    try:
                        with open(return_fifo, "w") as rf:
                            rf.write("1")
                    except:  # pylint: disable=bare-except
                        print("double failed...: ")
                        sys.stdout.flush()
                    else:
                        print(
                            "failed with {}: {}",
                            err.__class__.__name__,
                            getattr(err, "message", "<>"),
                        )
                        sys.stdout.flush()

                else:

                    # SEND THE SUCCESS RESPONSE
                    print("success...: ")
                    sys.stdout.flush()
                    with open(return_fifo, "w") as rf:
                        rf.write("0")

                    print("and wrote about it...: ")
                    sys.stdout.flush()


def main(test=False):  # pylint: disable=inconsistent-return-statements
    """
    read in the contents of the inputs yaml file
    """

    try:
        os.fork  # pylint: disable=no-member
    except NameError:
        raise RuntimeError(
            "scgrad.py depends on os.fork, which is not available on this system."
        )

    ARGS = sys.argv[1:]
    if ARGS[0] == "scgrad.py":
        ARGS.pop(0)

    WorkerDaemon = TestDaemon if test else GradientDaemon

    # --------------------------------------------------
    # Daemon controllers
    # --------------------------------------------------
    if ARGS[0] == "start":
        print("starting daemon")
        daemon = WorkerDaemon(DAEMON_PID, DAEMON_FIFO)
        daemon.start()
        return "0"

    if ARGS[0] == "stop":
        print("stopping daemon")
        daemon = WorkerDaemon(DAEMON_PID, DAEMON_FIFO)
        daemon.stop()
        return "0"

    if ARGS[0] == "restart":
        print("restaring daemon")
        daemon = WorkerDaemon(DAEMON_PID, DAEMON_FIFO)
        daemon.restart()
        return "0"

    # --------------------------------------------------
    # Gradient job
    # --------------------------------------------------

    try:
        with open(DAEMON_PID, "r") as pf:
            pid = int(pf.read().strip())
    except IOError:
        pid = None

    if not pid:
        raise RuntimeError("please start the daemon")

    assert (
        len(ARGS) == 4
    ), "ssc.py expects 4 parameters, including a commonfile, partfile, outfile and the component index"

    try:
        int(ARGS[3])
    except ValueError:
        raise ValueError("The args[3] must be an integer")

    # CREATE THE RESPONSE FIFO
    RETURN_FIFO = os.path.join("/tmp/sc-" + str(uuid.uuid4()) + ".fifo")
    os.mkfifo(RETURN_FIFO)  # pylint: disable=no-member

    def cleanup():
        os.remove(RETURN_FIFO)

    atexit.register(cleanup)

    # SEND THE ARGS TO THE DAEMON
    with open(DAEMON_FIFO, "w") as d_send:
        d_send.write(json.dumps(ARGS + [RETURN_FIFO]))

    # LISTEN FOR THE RESPONSE
    with open(RETURN_FIFO, "r") as d_return:
        response = d_return.read()
        print("daemon sent response: {}".format(response))
        return response


if __name__ == "__main__":
    print("hi from scgrad", sys.argv)
    condition_flag = main(test=False)
    if condition_flag == "0":
        print("Test Daemon worked!")
    else:
        print("Something went wrong: {}".format(condition_flag))
