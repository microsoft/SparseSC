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
    BATCH_DIRECTORY=BATCH_DIR,
)

run(my_config)

fitted_model = aggregate_batch_results("path/to/my/batch_config")

"""
# pylint: disable=differing-type-doc, differing-param-doc, missing-param-doc, missing-raises-doc, missing-return-doc
from __future__ import print_function
import pdb
import datetime
import io
import os
import sys
import time
import pathlib
import importlib
from typing import NamedTuple, Optional
import azure.storage.blob as azureblob
from azure.storage.blob.models import ContainerPermissions
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as models
from jsonschema import validate
from SparseSC.cli.stt import get_config
from .print_progress import print_progress

from yaml import load, dump

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

# pylint: disable=fixme

_STANDARD_OUT_FILE_NAME = "stdout.txt"  # Standard Output file

# pylint: disable=bad-continuation, invalid-name, protected-access, line-too-long

try:
    input = raw_input  # pylint:  disable=redefined-builtin
except NameError:
    pass


_CONTAINER_OUTPUT_FILE = "output.yaml"  # Standard Output file
_CONTAINER_INPUT_FILE = "input.yaml"  # Standard Output file

_BATCH_CV_FILE_NAME = "cv_parameters.yaml"

_GRAD_COMMON_FILE = "common.yaml"
_GRAD_PART_FILE = "part.yaml"

# _BATCH_FIT_FILE_NAME = "fit_parameters.yaml"

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
        datetime.datetime.utcnow() + datetime.timedelta(hours=1),
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
        container_url=container_sas_url, path="fold_{}.yaml".format(fold_number)
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


def upload_file_to_container(block_blob_client, container_name, file_path):
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
        expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2),
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
    container_conf = batch.models.ContainerConfiguration(
        container_image_names=["python:3.7", "jdthorpe/sparsesc:x-grad-daemon"]
    )
    new_pool = batch.models.PoolAddParameter(
        id=config.POOL_ID,
        virtual_machine_configuration=batch.models.VirtualMachineConfiguration(
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

    job_description = batch.models.JobAddParameter(
        id=job_id, pool_info=batch.models.PoolInformation(pool_id=pool_id)
    )

    batch_service_client.job.add(job_description)


def add_tasks(
    _blob_client, batch_service_client, container_sas_url, job_id, _input_file, count
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
            image_name="jdthorpe/sparsesc:x-grad-daemon",
            container_run_options="scgrad start",
        )

        tasks.append(
            batch.models.TaskAddParameter(
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
    # ptrn = re.compile(r"^fold_\d+.yaml$")
    ptrn = "fold_{}.yaml"

    pathlib.Path(config.BATCH_DIRECTORY).mkdir(parents=True, exist_ok=True)
    blob_names = [b.name for b in _blob_client.list_blobs(config.CONTAINER_NAME)]

    for i in range(count):
        blob_name = ptrn.format(i)
        if not blob_name in blob_names:
            raise RuntimeError("incomplete blob set: missing blob {}".format(blob_name))
        out_path = os.path.join(config.BATCH_DIRECTORY, blob_name)
        _blob_client.get_blob_to_path(config.CONTAINER_NAME, blob_name, out_path)


def _download_results(config, _blob_client, out_path, count, ptrn="fold_{}.yaml"):
    # ptrn = re.compile(r"^fold_\d+.yaml$")

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


# ------------------------------
# Fail Faster
# ------------------------------
config_schema = {
    "type": "object",
    "properties": {
        "BATCH_ACCOUNT_NAME": {"type": "string"},
        "BATCH_ACCOUNT_KEY": {"type": "string"},
        "BATCH_ACCOUNT_URL": {"type": "string"},
        "STORAGE_ACCOUNT_NAME": {"type": "string"},
        "STORAGE_ACCOUNT_KEY": {"type": "string"},
        "POOL_ID": {"type": "string"},
        "POOL_NODE_COUNT": {"type": "number", "minimum": 0},
        "POOL_LOW_PRIORITY_NODE_COUNT": {"type": "number", "minimum": 0},
        "POOL_VM_SIZE": {"type": "string"},
        "DELETE_POOL_WHEN_DONE": {"type": "boolean"},
        "JOB_ID": {"type": "string"},
        "DELETE_JOB_WHEN_DONE": {"type": "boolean"},
        "CONTAINER_NAME": {
            "type": "string",
            "pattern": "^[a-z0-9](-?[a-z0-9]+)$",
            "maxLength": 63,
            "minLength": 3,
        },
        "BATCH_DIRECTORY": {"type": "string"},
    },
    # TODO: missing required properties
}


class BatchConfig(NamedTuple):
    """
    A convenience class for typing the config object
    """

    # pylint: disable=too-few-public-methods
    POOL_ID: str
    JOB_ID: str
    POOL_VM_SIZE: str
    CONTAINER_NAME: str
    BATCH_DIRECTORY: str
    POOL_NODE_COUNT: int = 0
    POOL_LOW_PRIORITY_NODE_COUNT: int = 0
    DELETE_POOL_WHEN_DONE: bool = False
    DELETE_JOB_WHEN_DONE: bool = False
    BATCH_ACCOUNT_NAME: Optional[str] = None
    BATCH_ACCOUNT_KEY: Optional[str] = None
    BATCH_ACCOUNT_URL: Optional[str] = None
    STORAGE_ACCOUNT_NAME: Optional[str] = None
    STORAGE_ACCOUNT_KEY: Optional[str] = None


service_keys = (
    "BATCH_ACCOUNT_NAME",
    "BATCH_ACCOUNT_KEY",
    "BATCH_ACCOUNT_URL",
    "STORAGE_ACCOUNT_NAME",
    "STORAGE_ACCOUNT_KEY",
)

_env_config = {}
for key in service_keys:
    val = os.getenv(key, None)
    if val:
        _env_config[key] = val.strip('"')


def validate_config(config):
    """
    validate the batch configuration object
    """
    _config = config._asdict()
    for _key in service_keys:
        if not _config[_key]:
            del _config[_key]

    __env_config = _env_config.copy()
    __env_config.update(_config)
    validate(__env_config, config_schema)
    return BatchConfig(**__env_config)


def run(config: BatchConfig) -> None:
    """
    Run a batch job
    """
    # pylint: disable=too-many-locals

    # replace any missing values in the configuration with environment variables
    config = validate_config(config)

    start_time = datetime.datetime.now().replace(microsecond=0)

    print(
        'Synthetic Controls Run "{}" start time: {}'.format(config.JOB_ID, start_time)
    )
    print()

    _LOCAL_INPUT_FILE = os.path.join(config.BATCH_DIRECTORY, _BATCH_CV_FILE_NAME)

    pdb.set_trace()

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
        blob_client, config.CONTAINER_NAME, input_file_path
    )

    # Create a Batch service client. We'll now be interacting with the Batch
    # service in addition to Storage
    credentials = batch_auth.SharedKeyCredentials(
        config.BATCH_ACCOUNT_NAME, config.BATCH_ACCOUNT_KEY
    )

    batch_client = batch.BatchServiceClient(
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
            blob_client,
            batch_client,
            CONTAINER_SAS_URL,
            config.JOB_ID,
            input_file,
            n_folds,
        )

        # Pause execution until tasks reach Completed state.
        wait_for_tasks_to_complete(
            batch_client, config.JOB_ID, datetime.timedelta(hours=24)
        )

        _download_files(config, blob_client, config.BATCH_DIRECTORY, n_folds)
        # Print the stdout.txt and stderr.txt files for each task to the console
    # --         print_task_output(batch_client, config.JOB_ID)

    except models.BatchErrorException as err:
        print_batch_exception(err)
        raise

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


class gradient_batch_client:
    """
    Client object for performing gradient calculations with azure batch
    """

    def __init__(self, config: BatchConfig, common_data, K, verbose=True):

        # replace any missing values in the configuration with environment variables
        config = validate_config(config)

        self.config = config
        self.K = K

        self.blob_client = azureblob.BlockBlobService(
            account_name=config.STORAGE_ACCOUNT_NAME,
            account_key=config.STORAGE_ACCOUNT_KEY,
        )

        # Use the blob client to create the containers in Azure Storage if they
        # don't yet exist.
        self.blob_client.create_container(config.CONTAINER_NAME, fail_on_exist=False)
        self.CONTAINER_SAS_URL = build_output_sas_url(config, self.blob_client)

        # Create a Batch service client. We'll now be interacting with the Batch
        # service in addition to Storage
        self.credentials = batch_auth.SharedKeyCredentials(
            config.BATCH_ACCOUNT_NAME, config.BATCH_ACCOUNT_KEY
        )

        self.batch_client = batch.BatchServiceClient(
            self.credentials, batch_url=config.BATCH_ACCOUNT_URL
        )

        # Upload The common files.
        self.common_file = self.upload_object_to_container(
            self.blob_client, config.CONTAINER_NAME, _GRAD_COMMON_FILE, common_data
        )

        # Create the pool that will contain the compute nodes that will execute the
        # tasks.
        try:
            create_pool(self.config, self.batch_client)
            if verbose:
                print("Created pool: ", self.config.POOL_ID)
        except models.BatchErrorException:
            if verbose:
                print("Using pool: ", self.config.POOL_ID)

    def do_grad(self, part_data):  # , verbose=True
        """
        calculate the gradient
        """
        start_time = datetime.datetime.now().replace(microsecond=0)
        print("Gradient start time: {}".format(start_time))

        timestamp = datetime.datetime.utcnow().strftime("%H%M%S")
        JOB_ID = self.config.JOB_ID + timestamp
        try:

            # Upload the part file
            part_file = self.upload_object_to_container(
                self.blob_client, self.config.CONTAINER_NAME, _GRAD_PART_FILE, part_data
            )

            # Create the job that will run the tasks.
            create_job(self.batch_client, JOB_ID, self.config.POOL_ID)

            # Add the tasks to the job.
            self.add_tasks(part_file, JOB_ID)

            # Pause execution until tasks reach Completed state.
            print("wait_for_tasks_to_complete")
            wait_for_tasks_to_complete(
                self.batch_client, JOB_ID, datetime.timedelta(hours=24)
            )

            print("_download_results")
            results = _download_results(
                self.config,
                self.blob_client,
                self.config.BATCH_DIRECTORY,
                self.K,
                self.output_file_pattern,
            )
            print("_downloaded_results")

            # TODO: inspect tasks for out of memory and other errors

            if self.config.DELETE_JOB_WHEN_DONE:
                self.batch_client.job.delete(JOB_ID)

            return results

        except Exception as err:

            pdb.set_trace()
            raise RuntimeError(
                "something went wrong: {}({})".format(
                    err.__class__.__name__, getattr(err, "message", "")
                )
            )

    output_file_pattern = "grad_{}.yml"

    def add_tasks(self, part_file, JOB_ID):
        """
        Adds a task for each input file in the collection to the specified job.
        """
        # print("Adding {} tasks to job [{}]...".format(count, job_id))
        tasks = list()
        for i in range(self.K):
            output_file = self.build_output_file(i)
            command_line = "/bin/bash -c 'scgrad start && scgrad {} {} {} {}'".format(
                _GRAD_COMMON_FILE, _GRAD_PART_FILE, _CONTAINER_OUTPUT_FILE, i
            )

            task_container_settings = models.TaskContainerSettings(
                image_name="jdthorpe/sparsesc:x-grad-daemon"
            )

            tasks.append(
                batch.models.TaskAddParameter(
                    id="grad_part_{}".format(i),
                    command_line=command_line,
                    resource_files=[self.common_file, part_file],
                    output_files=[output_file],
                    container_settings=task_container_settings,
                )
            )

        pdb.set_trace()
        self.batch_client.task.add_collection(JOB_ID, [tasks[0]])

    def build_output_file(self, i):
        """
        Uploads a local file to an Azure Blob storage container.

        :rtype: `azure.batch.models.ResourceFile`
        :return: A ResourceFile initialized with a SAS URL appropriate for Batch
        tasks.
        """
        # where to store the outputs
        container_dest = models.OutputFileBlobContainerDestination(
            container_url=self.CONTAINER_SAS_URL,
            path=self.output_file_pattern.format(i),
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

    def upload_object_to_container(
        self, block_blob_client, container_name, blob_name, obj
    ):
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
        # print("Uploading file {} to container [{}]...".format(blob_name, container_name))

        block_blob_client.create_blob_from_text(
            container_name, blob_name, dump(obj, Dumper=Dumper)
        )

        sas_token = block_blob_client.generate_blob_shared_access_signature(
            container_name,
            blob_name,
            permission=azureblob.BlobPermissions.READ,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=2),
        )

        sas_url = block_blob_client.make_blob_url(
            container_name, blob_name, sas_token=sas_token
        )

        return models.ResourceFile(http_url=sas_url, file_path=blob_name)


if __name__ == "__main__":
    # TODO: this is not an ideal API
    config_module = importlib.__import__("config")
    run(config_module.config)
