"""
AzureBatch module
"""
from .azure_batch_client import gradient_batch_client, run
from .BatchConfig import BatchConfig
from .azure_batch_client import BatchConfig, gradient_batch_client
from .aggregate_results import aggregate_batch_results

