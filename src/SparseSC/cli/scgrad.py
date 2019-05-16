"""
somethings something
"""
# pylint: disable=invalid-name, unused-import
import sys
import numpy as np
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


pluck = lambda d, *args: (d[arg] for arg in args)


# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON
# AT THE VERY LEAST THIS NEEDS TO BE RUN AS A DAEMON

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


def main():
    """
    read in the contents of the inputs yaml file
    """
    ARGS = sys.argv[1:]
    if ARGS[0] == "scgrad.py":
        ARGS.pop(0)
    assert (
        len(ARGS) == 3
    ), "ssc.py expects 2 parameters, including a commonfile, partfile, outfile"

    commonfile, partfile, outfile = ARGS # pylint: disable=unbalanced-tuple-unpacking

    with open(commonfile, "r") as fp:
        common = load(fp, Loader=Loader)
    with open(partfile, "r") as fp:
        part = load(fp, Loader=Loader)

    grad = grad_part(common, part)

    with open(outfile, "w") as fp:
        fp.write(dump(grad, Dumper=Dumper))


if __name__ == "__main__":
    main()
