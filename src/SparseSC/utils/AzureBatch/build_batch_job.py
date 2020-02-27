"""
USAGE: 

python 
create_job()
"""
from __future__ import print_function
from os.path import join
import super_batch
from SparseSC.cli.stt import get_config
from .constants import (
    _CONTAINER_OUTPUT_FILE,
    _CONTAINER_INPUT_FILE,
    _BATCH_CV_FILE_NAME,
)

LOCAL_OUTPUTS_PATTERN = "fold_{}.yaml"


def create_job(client: super_batch.Client, batch_dir: str) -> None:
    r"""
    :param client: A :class:`super_batch.Client` instance with the Azure Batch run parameters
    :type client: :class:super_batch.Client

    :param str batch_dir: path of the local batch temp directory
    """
    _LOCAL_INPUT_FILE = join(batch_dir, _BATCH_CV_FILE_NAME)

    v_pen, w_pen, model_data = get_config(_LOCAL_INPUT_FILE)
    n_folds = len(model_data["folds"]) * len(v_pen) * len(w_pen)

    # CREATE THE COMMON IMPUT FILE RESOURCE
    input_resource = client.build_resource_file(
        _LOCAL_INPUT_FILE, _CONTAINER_INPUT_FILE
    )

    for fold_number in range(n_folds):

        # BUILD THE COMMAND LINE
        command_line = "/bin/bash -c 'stt {} {} {}'".format(
            _CONTAINER_INPUT_FILE, _CONTAINER_OUTPUT_FILE, fold_number
        )

        # CREATE AN OUTPUT RESOURCE:
        output_resource = client.build_output_file(
            _CONTAINER_OUTPUT_FILE, LOCAL_OUTPUTS_PATTERN.format(fold_number)
        )

        # CREATE A TASK
        client.add_task([input_resource], [output_resource], command_line=command_line)
