# Using Azure Batch from the Bash Terminal

## Setup

The azure batch client requires some additional dependencies which can be installed via:

```bash
pip install  azure-batch azure-storage-blob jsonschema pyyaml
```

# Createing Azure Resources

Using Azure batch requires an azure account, and we'll demonstrate how to run
this module using the [azure command line tool]().

After logging into the console with `az login`,  you'll need to create an azure
resource group into which the batch account is created.  In addition, the 
azure batch service requires a storage account which is used to keep track of
details of the jobs and tasks.

Althought the the resource group, storage account and batch account could have
different names, for sake of expositoin, we'll give them all the same name and
locate them in the US West 2 region, which we'll delare by creating variables
like so:

```bash
name="sparsesctest"
location="westus2"
```

Next, the required resource group, storage account and batch account can be
created with the following commands:

```bash
az group create -l $location -n $name
az storage account create -n $name -g run-dammit
az batch account create -l $location -n $name -g $name --storage-account $name
```

Finally, we'll need some information about created accounts in order to create
and run batch jobs. We can create bash variables that contain the information
that the SparseSC azure batch client will require, with the following:

```bash
export BATCH_ACCOUNT_NAME=$name
export BATCH_ACCOUNT_KEY=$(az batch account keys list -n $name -g $name --query primary)
export BATCH_ACCOUNT_URL="https://$name.$location.batch.azure.com"
export STORAGE_ACCOUNT_NAME=$name
export STORAGE_ACCOUNT_KEY=$(az storage account keys list -n $name --query [0].value)
```

We could of course echo these to the console and copy/paste the values into the
BatchConfig object below. however we don't need to do that if we run python
from within the same terminal session, as these variales will be found by the
SparseSC batch client if they are not provided explicitly.

# Prepare parameters for the Batch Job

Parameters for a batch job can be created using `fit()` by providing a directory where the batch parameters should be stored:
```python
from SparseSC import fit
batchdir = os.path.expanduser("/path/to/my/batch/data/")

fit(x, y, ... , batchdir = batchdir)
```

# Executing the Batch Job

In the following Python script, a Batch configuration is created and the batch
job is executed with Azure Batch. Note that the Batch Account and Storage
Account details can be provided directly to the Batch Config, with default
values taken from the system envoironment.

```python
import os
from datetime import datetime
from SparseSC.utils.azure_batch_client import BatchConfig, run as run_batch_job, aggregate_batch_results

name = "test43"
# Batch job names must be unique, and a 
timestamp = datetime.utcnow().strftime("%H%M%S")
batchdir = os.path.expanduser("/path/to/my/batch/data/")

my_config = BatchConfig(
	# Name of the VM pool
    POOL_ID= name,
	# number of standard nodes
    POOL_NODE_COUNT=5,
	# number of low priority nodes
    POOL_LOW_PRIORITY_NODE_COUNT=5,
	# VM type 
    POOL_VM_SIZE= "STANDARD_A1_v2",
	# Job ID.  Note that this must be unique.
    JOB_ID= name + timestamp,
	# Name of the storage container for storing parameters and results
    CONTAINER_NAME= name,
	# local directory with the parameters, and where the results will go
    BATCH_DIRECTORY= batchdir,
	)

# run the batch job
run_batch_job(my_config)

# aggregate the results into a fitted model instance
fitted_model = aggregate_batch_results(batchdir)
```

# Cleaning Up

In order to prevent unexpected charges, the resource group, including all the
resources it contains, such as the storge account and batch pools, can be
removed with the following command.

```bash
az group delete -n $name
```
