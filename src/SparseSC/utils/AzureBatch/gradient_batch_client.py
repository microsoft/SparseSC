"""
gradient level batching
"""
# pylint: disable=differing-type-doc, differing-param-doc, missing-param-doc, missing-raises-doc, missing-return-doc
from __future__ import print_function
import pdb
import datetime
import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batch_auth
import azure.batch.models as models
from .BatchConfig import BatchConfig, validate_config
# from .azure_batch_client import (
#     build_output_sas_url,
#     create_pool,
#     create_job,
#     wait_for_tasks_to_complete,
#     _download_results,
# )

from yaml import dump

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

from .constants import _CONTAINER_OUTPUT_FILE, _GRAD_COMMON_FILE, _GRAD_PART_FILE

OUTPUT_FILE_PATTERN = "grad_{}.yml"

class gradient_batch_client:
    """
    Client object for performing gradient calculations with azure batch
    """

    # pylint: disable=no-self-use

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
                OUTPUT_FILE_PATTERN,
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


    def add_tasks(self, part_file, JOB_ID):
        """
        Adds a task for each input file in the collection to the specified job.
        """
        # print("Adding {} tasks to job [{}]...".format(count, job_id))
        tasks = list()
        for i in range(self.K):
            output_file = self.build_output_file(i)
            command_line = "/bin/bash -c 'echo $AZ_BATCH_TASK_WORKING_DIR && daemon status && scgrad {} {} {} {}'".format(
                _GRAD_COMMON_FILE, _GRAD_PART_FILE, _CONTAINER_OUTPUT_FILE, i
            )

            if self.config.REGISTRY_USERNAME:
                registry = batch.models.ContainerRegistry(
                    user_name=self.config.REGISTRY_USERNAME,
                    password=self.config.REGISTRY_PASSWORD,
                    registry_server=self.config.REGISTRY_SERVER,
                )
                task_container_settings = models.TaskContainerSettings(
                    image_name=self.config.DOCKER_IMAGE, registry=registry
                )
                # pdb.set_trace()
            else:
                task_container_settings = models.TaskContainerSettings(
                    image_name=self.config.DOCKER_IMAGE
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
            path=OUTPUT_FILE_PATTERN.format(i),
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
            expiry=datetime.datetime.utcnow()
            + datetime.timedelta(hours=self.config.STORAGE_ACCESS_DURATION_HRS),
        )

        sas_url = block_blob_client.make_blob_url(
            container_name, blob_name, sas_token=sas_token
        )

        return models.ResourceFile(http_url=sas_url, file_path=blob_name)
