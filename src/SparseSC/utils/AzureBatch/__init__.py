"""
AzureBatch module
"""
from .azure_batch_client import BatchConfig, gradient_batch_client
from .aggregate_results import aggregate_batch_results


# CONTSTANTS USED BY THIS MODULE

_CONTAINER = "jathorpe/sparsesc"
_STANDARD_OUT_FILE_NAME = "stdout.txt"  # Standard Output file
_CONTAINER_OUTPUT_FILE = "output.yaml"  # Standard Output file
_CONTAINER_INPUT_FILE = "input.yaml"  # Standard Output file
_BATCH_CV_FILE_NAME = "cv_parameters.yaml"
_BATCH_FIT_FILE_NAME = "fit_parameters.yaml"
_GRAD_COMMON_FILE = "common.yaml"
_GRAD_PART_FILE = "part.yaml"



