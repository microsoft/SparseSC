"""
utilities for doing gradient descent in batches
"""
import os
import numpy as np
import subprocess


def single_grad_cli(
    tmpdir, N0, N1, in_controls, splits, b_i, w_pen, treated_units, Y_treated, Y_control
):
    """
    wrapper for the real function 
    """
    from yaml import load, dump

    try:
        from yaml import CLoader as Loader, CDumper as Dumper
    except ImportError:
        from yaml import Loader, Dumper

    _common_params = {
        "N0": N0,
        "N1": N1,
        "in_controls": in_controls,
        "splits": splits,
        "b_i": b_i,
        "w_pen": w_pen,
        "treated_units": treated_units,
        "Y_treated": Y_treated,
        "Y_control": Y_control,
    }

    COMMONFILE = os.path.join(tmpdir, "commonfile.yaml")
    PARTFILE = os.path.join(tmpdir, "partfile.yaml")
    OUTFILE = os.path.join(tmpdir, "outfile.yaml")

    with open(COMMONFILE, "w") as fp:
        fp.write(dump(_common_params, Dumper=Dumper))

    def inner(A, weights, dA_dV_ki_k, dB_dV_ki_k):
        """
        Calculate a single component of the gradient
        """
        _local_params = {
            "A": A,
            "weights": weights,
            "dA_dV_ki_k": dA_dV_ki_k,
            "dB_dV_ki_k": dB_dV_ki_k,
        }
        with open(PARTFILE, "w") as fp:
            fp.write(dump(_local_params, Dumper=Dumper))

        subprocess.run(["scgrad", COMMONFILE, PARTFILE, OUTFILE])

        with open(OUTFILE, "r") as fp:
            val = load(fp, Loader=Loader)
            return val

    return inner


def single_grad(
    N0, N1, in_controls, splits, b_i, w_pen, treated_units, Y_treated, Y_control
):
    """
    wrapper for the real function 
    """

    in_controls2 = [np.ix_(i, i) for i in in_controls]

    def inner(A, weights, dA_dV_ki_k, dB_dV_ki_k):
        """
        Calculate a single component of the gradient
        """
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

    return inner
