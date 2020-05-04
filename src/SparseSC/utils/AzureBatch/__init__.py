"""
AzureBatch module
"""
# from .gradient_batch_client import gradient_batch_client # abandon for now
DOCKER_IMAGE_NAME = "jdthorpe/sparsesc:latest"
from .aggregate_results import aggregate_batch_results
from .build_batch_job import create_job
