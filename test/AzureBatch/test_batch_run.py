# --------------------------------------------------------------------------------
# Programmer: Jason Thorpe
# Date    1/25/2019 3:34:02 PM
# Language:  Python (.py) Version 2.7 or 3.5
# Usage:
#
# Test all model types
#
#     \SpasrseSC > python -m unittest test/test_fit.py
#
# Test a specific model type (e.g. "prospective-restricted"):
#
#     \SpasrseSC > python -m unittest test.test_fit.TestFit.test_retrospective
#
# --------------------------------------------------------------------------------
# pylint: disable=multiple-imports, missing-docstring, no-self-use
"""
USAGE (CMD):

az login

set name=sparsescbatchtesting
set rgname=SparseSC-batch-testing
set BATCH_ACCOUNT_NAME=%name%
for /f %i in ('az batch account keys list -n %name% -g %rgname% --query primary') do @set BATCH_ACCOUNT_KEY=%i
for /f %i in ('az batch account show -n %name% -g %rgname% --query accountEndpoint') do @set BATCH_ACCOUNT_ENDPOINT=%i
for /f %i in ('az storage account keys list -n %name% --query [0].value') do @set STORAGE_ACCOUNT_KEY=%i
for /f %i in ('az storage account show-connection-string --name %name% --query connectionString') do @set STORAGE_ACCOUNT_CONNECTION_STRING=%i

# clean up the quotes
set BATCH_ACCOUNT_KEY=%BATCH_ACCOUNT_KEY:"=%
set BATCH_ACCOUNT_ENDPOINT=%BATCH_ACCOUNT_ENDPOINT:"=%
set STORAGE_ACCOUNT_KEY=%STORAGE_ACCOUNT_KEY:"=%
set STORAGE_ACCOUNT_CONNECTION_STRING=%STORAGE_ACCOUNT_CONNECTION_STRING:"=%

cd test\AzureBatch
rm -rf data
python test_batch_build.py
python test_batch_run.py
python test_batch_aggregate.py
"""

from __future__ import print_function  # for compatibility with python 2.7
import os, unittest, datetime
from os.path import join, realpath, dirname, exists
from super_batch import Client

from SparseSC.utils.AzureBatch import (
    DOCKER_IMAGE_NAME,
    create_job,
)


class TestFit(unittest.TestCase):
    def test_retrospective_no_wait(self):
        """
        test the no-wait and load_results API
        """

        name = os.getenv("name")
        if name is None:
            raise RuntimeError(
                "Please create an environment variable called 'name' as en the example docs"
            )
        batch_dir = join(dirname(realpath(__file__)), "data", "batchTest")
        assert exists(batch_dir), "Batch Directory '{}' does not exist".format(
            batch_dir
        )

        timestamp = datetime.datetime.utcnow().strftime("%H%M%S")

        batch_client = Client(
            POOL_ID=name,
            POOL_LOW_PRIORITY_NODE_COUNT=5,
            POOL_VM_SIZE="STANDARD_A1_v2",
            JOB_ID=name + timestamp,
            BLOB_CONTAINER_NAME=name,
            BATCH_DIRECTORY=batch_dir,
            DOCKER_IMAGE=DOCKER_IMAGE_NAME,
        )
        create_job(batch_client, batch_dir)

        batch_client.run(wait=False)
        batch_client.load_results()

    def test_retrospective(self):

        name = os.getenv("name")
        if name is None:
            raise RuntimeError(
                "Please create an environment variable called 'name' as en the example docs"
            )
        batch_dir = join(dirname(realpath(__file__)), "data", "batchTest")
        assert exists(batch_dir), "Batch Directory '{}' does not exist".format(
            batch_dir
        )

        timestamp = datetime.datetime.utcnow().strftime("%H%M%S")

        batch_client = Client(
            POOL_ID=name,
            POOL_LOW_PRIORITY_NODE_COUNT=5,
            POOL_VM_SIZE="STANDARD_A1_v2",
            JOB_ID=name + timestamp,
            BLOB_CONTAINER_NAME=name,
            BATCH_DIRECTORY=batch_dir,
            DOCKER_IMAGE=DOCKER_IMAGE_NAME,
        )

        create_job(batch_client, batch_dir)
        batch_client.run()


if __name__ == "__main__":
    unittest.main()
