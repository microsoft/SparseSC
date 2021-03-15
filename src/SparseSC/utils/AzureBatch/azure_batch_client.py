"""
usage requires these additional modules

pip install azure-batch azure-storage-blob jsonschema pyyaml && pip install git+https://github.com/microsoft/SparseSC.git@ad4bf27edb28f517508f6934f21eb65d17fb6543 && scgrad start


usage:

from SparseSC import fit, aggregate_batch_results
from SparseSC.utils.azure_batch_client import BatchConfig, run

_TIMESTAMP = datetime.utcnow().strftime("%Y%m%d%H%M%S")

BATCH_DIR= "path/to/my/batch_config/"

fit(x=x,..., batchDir=BATCH_DIR)

my_config = BatchConfig(
    BATCH_ACCOUNT_NAME="MySecret",
    BATCH_ACCOUNT_KEY="MySecret",
    BATCH_ACCOUNT_URL="MySecret",
    STORAGE_ACCOUNT_NAME="MySecret",
    STORAGE_ACCOUNT_KEY="MySecret",
    POOL_ID="my-compute-pool",
    POOL_NODE_COUNT=0,
    POOL_LOW_PRIORITY_NODE_COUNT=20,
    POOL_VM_SIZE="STANDARD_A1_v2",
    DELETE_POOL_WHEN_DONE=False,
    JOB_ID="my-job" + _TIMESTAMP,
    DELETE_JOB_WHEN_DONE=False,
    CONTAINER_NAME="my-blob-container",
    BATCH_DIRECTORY=BATCH_DIR)

run(my_config)

fitted_model = aggregate_batch_results("path/to/my/batch_config")

"""
# pylint: disable=differing-type-doc, differing-param-doc, missing-param-doc, missing-raises-doc, missing-return-doc
from __future__ import print_function
import datetime
import io
import os
import sys
import time
import pathlib
import importlib
from collections import defaultdict
import azure.storage.blob as azureblob
from azure.storage.blob import ContainerSasPermissions as ContainerPermissions
from azure.batch import BatchServiceClient
import azure.batch.batch_auth as batch_auth
import azure.batch.models as models
from SparseSC.cli.stt import get_config
from ..print_progress import print_progress
from super_batch import BatchConfig

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from .constants import (
    _STANDARD_OUT_FILE_NAME,
    _CONTAINER_OUTPUT_FILE,
    _CONTAINER_INPUT_FILE,
    _BATCH_CV_FILE_NAME,
)

FOLD_FILE_PATTERN = "fold_{}.yaml"
# pylint: disable=bad-continuation, invalid-name, protected-access, line-too-long, fixme


sys.path.append(".")
sys.path.append("..")

# Update the Batch and Storage account credential strings in config.py with values
# unique to your accounts. These are used when constructing connection strings
# for the Batch and Storage client objects.
def build_output_sas_url(config, _blob_client):
    """
    build a sas token for the output container
    """

    sas_token = _blob_client.generate_container_shared_access_signature(
        config.CONTAINER_NAME,
        ContainerPermissions.READ
        + ContainerPermissions.WRITE
        + ContainerPermissions.DELETE
        + ContainerPermissions.LIST,
        datetime.datetime.utcnow() + datetime.timedelta(hours=config.STORAGE_ACCESS_DURATION_HRS),
        start=datetime.datetime.utcnow(),
    )

    _sas_url = "https://{}.blob.core.windows.net/{}?{}".format(
        config.STORAGE_ACCOUNT_NAME, config.CONTAINER_NAME, sas_token
    )
    return _sas_url


def print_batch_exception(batch_exception):
    """
    Prints the contents of the specified Batch exception.

    :param batch_exception:
    """
    print("-------------------------------------------")
    print("Exception encountered:")
    if (
        batch_exception.error
        and batch_exception.error.message
        and batch_exception.error.message.value
    ):
        print(batch_exception.error.message.value)
        if batch_exception.error.values:
            print()
            for mesg in batch_exception.error.values:
                print("{}:\t{}".format(mesg.key, mesg.value))
    print("-------------------------------------------")


def build_output_file(container_sas_url, fold_number):
    """
    Uploads a local file to an Azure Blob storage container.

    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
        tasks.
    """
    # where to store the outputs
    container_dest = models.OutputFileBlobContainerDestination(
        container_url=container_sas_url, path=FOLD_FILE_PATTERN.format(fold_number)
    )
    dest = models.OutputFileDestination(container=container_dest)

    # under what conditions should you attempt to extract the outputs?
    upload_options = models.OutputFileUploadOptions(
        upload_condition=models.OutputFileUploadCondition.task_success
    )

    # https://docs.microsoft.com/en-us/azure/batch/batch-task-output-files#specify-output-files-for-task-output
    return models.OutputFile(
        file_pattern=_CONTAINER_OUTPUT_FILE,
        destination=dest,
        upload_options=upload_options,
    )


def upload_file_to_container(block_blob_client, container_name, file_path, duration_hours=24):
    """
    Uploads a local file to an Azure Blob storage container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param str file_path: The local path to the file.
    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
        tasks.
    """
    blob_name = os.path.basename(file_path)

    print("Uploading file {} to container [{}]...".format(file_path, container_name))

    block_blob_client.create_blob_from_path(container_name, blob_name, file_path)

    sas_token = block_blob_client.generate_blob_shared_access_signature(
        container_name,
        blob_name,
        permission=azureblob.BlobPermissions.READ,
        expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=duration_hours),
    )

    sas_url = block_blob_client.make_blob_url(
        container_name, blob_name, sas_token=sas_token
    )

    return models.ResourceFile(http_url=sas_url, file_path=_CONTAINER_INPUT_FILE)


def create_pool(config, batch_service_client):
    """
    Creates a pool of compute nodes with the specified OS settings.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str pool_id: An ID for the new pool.
    :param str publisher: Marketplace image publisher
    :param str offer: Marketplace image offer
    :param str sku: Marketplace image sku
    """
    # Create a new pool of Linux compute nodes using an Azure Virtual Machines
    # Marketplace image. For more information about creating pools of Linux
    # nodes, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/
    image_ref_to_use = models.ImageReference(
        publisher="microsoft-azure-batch",
        offer="ubuntu-server-container",
        sku="16-04-lts",
        version="latest",
    )

    if config.REGISTRY_USERNAME:
        registry = models.ContainerRegistry(
            user_name=config.REGISTRY_USERNAME,
            password=config.REGISTRY_PASSWORD,
            registry_server=config.REGISTRY_SERVER,
        )
        container_conf = models.ContainerConfiguration(
            container_image_names=[config.DOCKER_CONTAINER],
            container_registries=[registry],
        )
    else:
        container_conf = models.ContainerConfiguration(
            container_image_names=[config.DOCKER_CONTAINER]
        )

    new_pool = models.PoolAddParameter(
        id=config.POOL_ID,
        virtual_machine_configuration=models.VirtualMachineConfiguration(
            image_reference=image_ref_to_use,
            container_configuration=container_conf,
            node_agent_sku_id="batch.node.ubuntu 16.04",
        ),
        vm_size=config.POOL_VM_SIZE,
        target_dedicated_nodes=config.POOL_NODE_COUNT,
        target_low_priority_nodes=config.POOL_LOW_PRIORITY_NODE_COUNT,
    )
    batch_service_client.pool.add(new_pool)


def create_job(batch_service_client, job_id, pool_id):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print("Creating job [{}]...".format(job_id))

    job_description = models.JobAddParameter(
        id=job_id, pool_info=models.PoolInformation(pool_id=pool_id)
    )

    batch_service_client.job.add(job_description)


def add_tasks(
    config,
    _blob_client,
    batch_service_client,
    container_sas_url,
    job_id,
    _input_file,
    count,
):
    """
    Adds a task for each input file in the collection to the specified job.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID of the job to which to add the tasks.
    :param list input_files: The input files
    :param output_container_sas_token: A SAS token granting write access to
        the specified Azure Blob storage container.
    """

    print("Adding {} tasks to job [{}]...".format(count, job_id))

    tasks = list()

    for fold_number in range(count):
        output_file = build_output_file(container_sas_url, fold_number)
        # command_line = '/bin/bash -c \'echo "Hello World" && echo "hello: world" > output.yaml\''
        command_line = "/bin/bash -c 'stt {} {} {}'".format(
            _CONTAINER_INPUT_FILE, _CONTAINER_OUTPUT_FILE, fold_number
        )

        task_container_settings = models.TaskContainerSettings(
            image_name=config.DOCKER_CONTAINER
        )

        tasks.append(
            models.TaskAddParameter(
                id="Task_{}".format(fold_number),
                command_line=command_line,
                resource_files=[_input_file],
                output_files=[output_file],
                container_settings=task_container_settings,
            )
        )

    batch_service_client.task.add_collection(job_id, tasks)


def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
    """
    Returns when all tasks in the specified job reach the Completed state.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The id of the job whose tasks should be to monitored.
    :param timedelta timeout: The duration to wait for task completion. If all
        tasks in the specified job do not reach Completed state within this time
        period, an exception will be raised.
    """

    _start_time = datetime.datetime.now()
    timeout_expiration = _start_time + timeout

    # print( "Monitoring all tasks for 'Completed' state, timeout in {}...".format(timeout), end="",)

    while datetime.datetime.now() < timeout_expiration:
        sys.stdout.flush()
        tasks = [t for t in batch_service_client.task.list(job_id)]

        incomplete_tasks = [
            task for task in tasks if task.state != models.TaskState.completed
        ]

        hours, remainder = divmod((datetime.datetime.now() - _start_time).seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print_progress(
            len(tasks) - len(incomplete_tasks),
            len(tasks),
            prefix="Time elapsed {:02}:{:02}:{:02}".format(
                int(hours), int(minutes), int(seconds)
            ),
            decimals=1,
            bar_length=min(len(tasks), 50),
        )

        error_codes = [t.execution_info.exit_code for t in tasks if t.execution_info and t.execution_info.exit_code ]
        if error_codes:
            codes = defaultdict(lambda : 0)
            for cd in error_codes:
                codes[cd] +=1 
            # import pdb; pdb.set_trace()
            raise RuntimeError( "\nSome tasks have exited with a non-zero exit code including: " + ", ".join([ "{}({})".format(k,v) for k, v in codes.items() ] ))
        if not incomplete_tasks:
            print()
            return True
        time.sleep(1)

    print()
    raise RuntimeError(
        "ERROR: Tasks did not reach 'Completed' state within "
        "timeout period of " + str(timeout)
    )


def print_task_output(batch_service_client, job_id, encoding=None):
    """Prints the stdout.txt file for each task in the job.

    :param batch_client: The batch client to use.
    :type batch_client: `batchserviceclient.BatchServiceClient`
    :param str job_id: The id of the job with task output files to print.
    """

    print("Printing task output...")

    tasks = batch_service_client.task.list(job_id)

    for task in tasks:

        node_id = batch_service_client.task.get(job_id, task.id).node_info.node_id
        print("Task: {}".format(task.id))
        print("Node: {}".format(node_id))

        stream = batch_service_client.file.get_from_task(
            job_id, task.id, _STANDARD_OUT_FILE_NAME
        )

        file_text = _read_stream_as_string(stream, encoding)
        print("Standard output:")
        print(file_text)


def _read_stream_as_string(stream, encoding):
    """Read stream as string

    :param stream: input stream generator
    :param str encoding: The encoding of the file. The default is utf-8.
    :return: The file content.
    :rtype: str
    """
    output = io.BytesIO()
    try:
        for data in stream:
            output.write(data)
        if encoding is None:
            encoding = "utf-8"
        return output.getvalue().decode(encoding)
    finally:
        output.close()
    raise RuntimeError("could not write data to stream or decode bytes")


def _download_files(config, _blob_client, out_path, count):

    pathlib.Path(config.BATCH_DIRECTORY).mkdir(parents=True, exist_ok=True)
    blob_names = [b.name for b in _blob_client.list_blobs(config.CONTAINER_NAME)]

    for i in range(count):
        blob_name = FOLD_FILE_PATTERN.format(i)
        if not blob_name in blob_names:
            raise RuntimeError("incomplete blob set: missing blob {}".format(blob_name))
        out_path = os.path.join(config.BATCH_DIRECTORY, blob_name)
        _blob_client.get_blob_to_path(config.CONTAINER_NAME, blob_name, out_path)


def _download_results(config, _blob_client, out_path, count, ptrn=FOLD_FILE_PATTERN):

    pathlib.Path(config.BATCH_DIRECTORY).mkdir(parents=True, exist_ok=True)
    blob_names = [b.name for b in _blob_client.list_blobs(config.CONTAINER_NAME)]

    results = []
    for i in range(count):
        blob_name = ptrn.format(i)
        if not blob_name in blob_names:
            raise RuntimeError("incomplete blob set: missing blob {}".format(blob_name))
        out_path = os.path.join(config.BATCH_DIRECTORY, blob_name)
        with _blob_client.get_blob_to_stream(
            config.CONTAINER_NAME, blob_name, out_path
        ) as blob:
            results[i] = load(blob, Loader=Loader)
    return results


def run(config: BatchConfig, wait=True) -> None:
    r"""
    :param config: A :class:`BatchConfig` instance with the Azure Batch run parameters
    :type config: :class:BatchConfig

    :param boolean wait: If true, wait for the batch to complete and then
            download the results to file

    :raises BatchErrorException: If raised by the Azure Batch Python SDK
    """
    # pylint: disable=too-many-locals

    start_time = datetime.datetime.now().replace(microsecond=0)

    print(
        'Synthetic Controls Run "{}" start time: {}'.format(config.JOB_ID, start_time)
    )
    print()

    _LOCAL_INPUT_FILE = os.path.join(config.BATCH_DIRECTORY, _BATCH_CV_FILE_NAME)

    v_pen, w_pen, model_data = get_config(_LOCAL_INPUT_FILE)
    n_folds = len(model_data["folds"]) * len(v_pen) * len(w_pen)

    # Create the blob client, for use in obtaining references to
    # blob storage containers and uploading files to containers.

    blob_client = azureblob.BlockBlobService(
        account_name=config.STORAGE_ACCOUNT_NAME, account_key=config.STORAGE_ACCOUNT_KEY
    )

    # Use the blob client to create the containers in Azure Storage if they
    # don't yet exist.
    blob_client.create_container(config.CONTAINER_NAME, fail_on_exist=False)
    CONTAINER_SAS_URL = build_output_sas_url(config, blob_client)

    # The collection of data files that are to be processed by the tasks.
    input_file_path = os.path.join(sys.path[0], _LOCAL_INPUT_FILE)

    # Upload the data files.
    input_file = upload_file_to_container(
        blob_client, config.CONTAINER_NAME, input_file_path, config.STORAGE_ACCESS_DURATION_HRS
    )

    # Create a Batch service client. We'll now be interacting with the Batch
    # service in addition to Storage
    credentials = batch_auth.SharedKeyCredentials(
        config.BATCH_ACCOUNT_NAME, config.BATCH_ACCOUNT_KEY
    )

    batch_client = BatchServiceClient(
        credentials, batch_url=config.BATCH_ACCOUNT_URL
    )

    try:
        # Create the pool that will contain the compute nodes that will execute the
        # tasks.
        try:
            create_pool(config, batch_client)
            print("Created pool: ", config.POOL_ID)
        except models.BatchErrorException:
            print("Using pool: ", config.POOL_ID)

        # Create the job that will run the tasks.
        create_job(batch_client, config.JOB_ID, config.POOL_ID)

        # Add the tasks to the job.
        add_tasks(
            config,
            blob_client,
            batch_client,
            CONTAINER_SAS_URL,
            config.JOB_ID,
            input_file,
            n_folds,
        )

        if not wait:
            return

        # Pause execution until tasks reach Completed state.
        wait_for_tasks_to_complete(
            batch_client, config.JOB_ID, datetime.timedelta(hours=config.STORAGE_ACCESS_DURATION_HRS)
        )

        _download_files(config, blob_client, config.BATCH_DIRECTORY, n_folds)

    except models.BatchErrorException as err:
        print_batch_exception(err)
        raise err

    # Clean up storage resources
    # TODO: re-enable this and delete the output container too
    # --     print("Deleting container [{}]...".format(input_container_name))
    # --     blob_client.delete_container(input_container_name)

    # Print out some timing info
    end_time = datetime.datetime.now().replace(microsecond=0)
    print()
    print("Sample end: {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))
    print()

    # Clean up Batch resources (if the user so chooses).
    if config.DELETE_POOL_WHEN_DONE:
        batch_client.pool.delete(config.POOL_ID)
    if config.DELETE_JOB_WHEN_DONE:
        batch_client.job.delete(config.JOB_ID)


def load_results(config: BatchConfig) -> None:
    r"""
    :param config: A :class:`BatchConfig` instance with the Azure Batch run parameters
    :type config: :class:BatchConfig

    :raises BatchErrorException: If raised by the Azure Batch Python SDK
    """
    # pylint: disable=too-many-locals

    start_time = datetime.datetime.now().replace(microsecond=0)
    print('Load result for job "{}" start time: {}'.format(config.JOB_ID, start_time))
    print()

    _LOCAL_INPUT_FILE = os.path.join(config.BATCH_DIRECTORY, _BATCH_CV_FILE_NAME)

    v_pen, w_pen, model_data = get_config(_LOCAL_INPUT_FILE)
    n_folds = len(model_data["folds"]) * len(v_pen) * len(w_pen)

    # Create the blob client, for use in obtaining references to
    # blob storage containers and uploading files to containers.

    blob_client = azureblob.BlockBlobService(
        account_name=config.STORAGE_ACCOUNT_NAME, account_key=config.STORAGE_ACCOUNT_KEY
    )

    # Create a Batch service client. We'll now be interacting with the Batch
    # service in addition to Storage
    credentials = batch_auth.SharedKeyCredentials(
        config.BATCH_ACCOUNT_NAME, config.BATCH_ACCOUNT_KEY
    )

    batch_client = BatchServiceClient(
        credentials, batch_url=config.BATCH_ACCOUNT_URL
    )

    try:

        # Pause execution until tasks reach Completed state.
        wait_for_tasks_to_complete(
            batch_client, config.JOB_ID, datetime.timedelta(hours=config.STORAGE_ACCESS_DURATION_HRS)
        )

        _download_files(config, blob_client, config.BATCH_DIRECTORY, n_folds)

    except models.BatchErrorException as err:
        print_batch_exception(err)
        raise err

    # Clean up storage resources
    # TODO: re-enable this and delete the output container too
    # --     print("Deleting container [{}]...".format(input_container_name))
    # --     blob_client.delete_container(input_container_name)

    # Print out some timing info
    end_time = datetime.datetime.now().replace(microsecond=0)
    print()
    print("Sample end: {}".format(end_time))
    print("Elapsed time: {}".format(end_time - start_time))
    print()

    # Clean up Batch resources (if the user so chooses).
    if config.DELETE_POOL_WHEN_DONE:
        batch_client.pool.delete(config.POOL_ID)
    if config.DELETE_JOB_WHEN_DONE:
        batch_client.job.delete(config.JOB_ID)


if __name__ == "__main__":
    # TODO: this is not an ideal API
    config_module = importlib.__import__("config")
    run(config_module.config)
