"""
somethings something
"""
# pylint: disable=invalid-name, unused-import
import sys
import numpy
from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from SparseSC.cross_validation import score_train_test


def get_config(infile):
    """
    read in the contents of the inputs yaml file
    """

    with open(infile, "r") as fp:
        config = load(fp, Loader=Loader)
    try:
        v_pen = tuple(config["v_pen"])
    except TypeError:
        v_pen = (config["v_pen"],)

    try:
        w_pen = tuple(config["w_pen"])
    except TypeError:
        w_pen = (config["w_pen"],)

    return v_pen, w_pen, config


def main():
    # GET THE COMMAND LINE ARGS
    ARGS = sys.argv[1:]
    if ARGS[0] == "ssc.py":
        ARGS.pop(0)
    assert (
        len(ARGS) == 3
    ), "ssc.py expects 2 parameters, including a file name and a batch number"
    infile, outfile, batchNumber = ARGS
    batchNumber = int(batchNumber)

    v_pen, w_pen, config = get_config(infile)
    n_folds = len(config["folds"]) * len(v_pen) * len(w_pen)

    assert 0 <= batchNumber < n_folds, "Batch number out of range"
    i_fold = batchNumber % len(config["folds"])
    i_v = (batchNumber // len(config["folds"])) % len(v_pen)
    i_w = (batchNumber // len(config["folds"])) // len(v_pen)

    params = config.copy()
    del params["folds"]
    del params["v_pen"]
    del params["w_pen"]

    train, test = config["folds"][i_fold]
    out = score_train_test(
        train=train, test=test, v_pen=v_pen[i_v], w_pen=w_pen[i_w], **params
    )

    with open(outfile, "w") as fp:
        fp.write(
            dump(
                {
                    "batch": batchNumber,
                    "i_fold": i_fold,
                    "i_v": i_v,
                    "i_w": i_w,
                    "results": out,
                },
                Dumper=Dumper,
            )
        )


if __name__ == "__main__":
    main()
