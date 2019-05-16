#!/usr/bin/env python
"""
A service like daemon for calculating components of the gradient
"""
# pylint: disable=invalid-name, unused-import, multiple-imports
import numpy as np
import uuid
import sys, time, os, atexit, json
from .daemon import Daemon
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

pluck = lambda d, *args: (d[arg] for arg in args)


def grad_part(common, part):
    """
    Calculate a single component of the gradient
    """
    N0, N1, in_controls, splits, b_i, w_pen, treated_units, Y_treated, Y_control = pluck(
        common,
        "N0",
        "N1",
        "in_controls",
        "splits",
        "b_i",
        "w_pen",
        "treated_units",
        "Y_treated",
        "Y_control",
    )

    in_controls2 = [np.ix_(i, i) for i in in_controls]

    A, weights, dA_dV_ki_k, dB_dV_ki_k = pluck(
        part, "A", "weights", "dA_dV_ki_k", "dB_dV_ki_k"
    )

    dPI_dV = np.zeros((N0, N1))  # stupid notation: PI = W.T
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


DAEMON_FIFO = "/var/sc-daemon.fifo"
DAEMON_PID = "/tmp/sc-gradient-daemon.pid"


class GradientDaemon(Daemon):
    """
    A daemon which calculates Sparse SC gradient components
    """
    def start(self):
        super().start()

        # SET UP THE FIFO
        os.mkfifo(DAEMON_FIFO)  # pylint: disable=no-member

        def cleanup():
            os.remove(DAEMON_FIFO)

        atexit.register(cleanup)

    def run(self):
        while True:
            with open(DAEMON_FIFO, "r") as fifo:
                try:
                    common_file, part_file, out_file, return_fifo = json.loads(
                        fifo.read()
                    )

                    # LOAD IN THE INPUT FILES
                    with open(common_file, "r") as fp:
                        common = load(fp, Loader=Loader)
                    with open(part_file, "r") as fp:
                        part = load(fp, Loader=Loader)

                    # DO THE WORK
                    grad = grad_part(common, part)

                    # DUMP THE RESULT TO THE OUTPUT FILE
                    with open(out_file, "w") as fp:
                        fp.write(dump(grad, Dumper=Dumper))

                except:  # pylint: disable=bare-except

                    # SOMETHING WENT WRONG, RESPOND WITH A NON-ZERO 
                    try:
                        with open(return_fifo, "w" + nonbloking) as rf:
                            rf.write("1")
                    except:  # pylint: disable=bare-except
                        pass

                else:

                    # SEND THE SUCCESS RESPONSE
                    with open(return_fifo, "w" + nonbloking) as rf:
                        rf.write("0")



def main(): # pylint: disable=inconsistent-return-statements
    """
    read in the contents of the inputs yaml file
    """

    try:
        os.fork # pylint: disable=no-member
    except NameError:
        raise RuntimeError("scgrad.py depends on os.fork, which is not available on this system.")

    try:
        with open(DAEMON_PID, "r") as pf:
            pid = int(pf.read().strip())
    except IOError:
        pid = None

    if not pid:
        daemon = GradientDaemon(DAEMON_PID)
        daemon.start()

    ARGS = sys.argv[1:]
    if ARGS[0] == "scgrad.py":
        ARGS.pop(0)

    if ARGS[0] == "start":
        daemon = GradientDaemon(DAEMON_PID)
        daemon.start()
        return

    if ARGS[0] == "stop":
        daemon = GradientDaemon(DAEMON_PID)
        daemon.stop()
        return

    if ARGS[0] == "restart":
        daemon = GradientDaemon(DAEMON_PID)
        daemon.restart()
        return

    assert (
        len(ARGS) == 3
    ), "ssc.py expects 2 parameters, including a commonfile, partfile, outfile"

    # CREATE THE RESPONSE FIFO
    RETURN_FIFO = os.path.join("/tmp/sc-" + str(uuid.uuid4()) + ".fifo")
    os.mkfifo(RETURN_FIFO)  # pylint: disable=no-member

    def cleanup():
        os.remove(RETURN_FIFO)

    atexit.register(cleanup)

    # SEND THE ARGS TO THE DAEMON
    with open(DAEMON_FIFO, "w") as d:
        d.write(json.dumps(ARGS + [RETURN_FIFO]))

    # LISTEN FOR THE RESPONSE
    with open(RETURN_FIFO, "r") as d:
        return d.read()


if __name__ == "__main__":
    condition_flag = main()
    if condition_flag == "0":
        print("Gradient calculated!")
    else:
        print("Something went wrong")
