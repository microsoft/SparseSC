"""
for local testing of the daemon 
"""
# pylint: disable=differing-type-doc, differing-param-doc, missing-param-doc, missing-raises-doc, missing-return-doc
import datetime
import os
import json
import tempfile
import atexit
import subprocess
import itertools
import tarfile
import io

import numpy as np
from ..cli.scgrad import GradientDaemon, DAEMON_PID, DAEMON_FIFO

from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# pylint: disable=fixme, too-few-public-methods
# pylint: disable=bad-continuation, invalid-name, protected-access, line-too-long

try:
    input = raw_input  # pylint:  disable=redefined-builtin
except NameError:
    pass


_CONTAINER_OUTPUT_FILE = "output.yaml"  # Standard Output file
_GRAD_COMMON_FILE = "common.yaml"
_GRAD_PART_FILE = "part.yaml"

RETURN_FIFO = "/tmp/sc-return.fifo"
os.mkfifo(RETURN_FIFO)  # pylint: disable=no-member


def cleanup():
    """ clean up"""
    os.remove(RETURN_FIFO)


atexit.register(cleanup)


class local_batch_daemon:
    """
    Client object for performing gradient calculations with azure batch
    """

    def __init__(self, common_data, K):
        subprocess.call(["python", "-m", "SparseSC.cli.scgrad", "start"])
        # CREATE THE RESPONSE FIFO
        # replace any missing values with environment variables
        self.common_data = common_data
        self.K = K

        # BUILT THE TEMPORARY FILE NAMES
        self.tmpDirManager = tempfile.TemporaryDirectory()
        self.tmpdirname = self.tmpDirManager.name
        print("Created temporary directory:", self.tmpdirname)
        self.GRAD_PART_FILE = os.path.join(self.tmpdirname, _GRAD_PART_FILE)
        self.CONTAINER_OUTPUT_FILE = os.path.join(self.tmpdirname, _CONTAINER_OUTPUT_FILE)

        # WRITE THE COMMON DATA TO FILE:
        with open(os.path.join(self.tmpdirname, _GRAD_COMMON_FILE), "w") as fh:
            fh.write(dump(self.common_data, Dumper=Dumper))


#--         # A UTILITY FUNCTION
#--         def tarify(x,name):
#--             with tarfile.open(os.path.join(self.tmpdirname, '{}.tar.gz'.format(name)), mode='w:gz') as dest_file:
#--                 for i, k in itertools.product( range(len(x)), range(len(x[0]))):
#--                     fname = 'arr_{}_{}'.format(i,k)
#--                     array_bytes = x[i][k].tobytes()
#--                     info = tarfile.TarInfo(fname)
#--                     info.size = len(array_bytes)
#--                     dest_file.addfile(info,io.BytesIO(array_bytes) )
#-- 
#--         tarify(part_data["dA_dV_ki"],"dA_dV_ki") 
#--         tarify(part_data["dB_dV_ki"],"dB_dV_ki") 
#--         import pdb; pdb.set_trace()


    def stop(self):
        """
        stop the daemon
        """
        # pylint: disable=no-self-use
        subprocess.call(["python", "-m", "SparseSC.cli.scgrad", "stop"])

    def do_grad(self, part_data):
        """
        calculate the gradient
        """
        start_time = datetime.datetime.now().replace(microsecond=0)
        print("Gradient start time: {}".format(start_time))

        # WRITE THE PART DATA TO FILE
        with open(self.GRAD_PART_FILE, "w") as fh:
            fh.write(dump(part_data, Dumper=Dumper))

        print("Gradient A")
#--         for key in part_data.keys():
#--             with open(os.path.join(self.tmpdirname, key), "w") as fh:
#--                 fh.write(dump(part_data[key], Dumper=Dumper))
#--         with open(os.path.join(self.tmpdirname, "item0.yaml"), "w") as fh: fh.write(dump(part_data["dA_dV_ki"][0][0], Dumper=Dumper))
#--         with open(os.path.join(self.tmpdirname, "item0.bytes"), "wb") as fh: fh.write(part_data["dA_dV_ki"][0][0].tobytes())
#--         import gzip
#-- 
#--         with gzip.open(os.path.join(self.tmpdirname, "item0_{}.gz".format(i)), "rb") as fh: matbytes = fh.read()

        dGamma0_dV_term2 = np.zeros(self.K)
        for k in range(self.K):
            print(k, end=" ")
            # SEND THE ARGS TO THE DAEMON
            with open(DAEMON_FIFO, "w") as df:
                df.write(
                    json.dumps(
                        [
                            self.tmpdirname,
                            RETURN_FIFO,
                            k,
                        ]
                    )
                )

            # LISTEN FOR THE RESPONSE
            with open(RETURN_FIFO, "r") as rf:
                response = rf.read()
            if response != "0":
                raise RuntimeError("Something went wrong in the daemon: {}".format( response))

            with open(self.CONTAINER_OUTPUT_FILE, "r") as fh:
                dGamma0_dV_term2[k] = load(fh.read(), Loader=Loader)

        return dGamma0_dV_term2

