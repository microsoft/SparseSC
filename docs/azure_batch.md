# Running Jobs in Parallel with Azure Batch 

Fitting a Sparse Synthetic Controls model can result in a very long running
time. Fortunately much of the work can be done in parallel and executed in
the cloud, and the SparseSC package comes with an Azure Batch utility which
can be used to fit a Synthetic controls model using Azure Batch.  There are
code examples for Windows CMD, Bash and Powershell.

## Setup

The azure batch client requires some additional dependencies which can be installed via:
```bash
pip install azure-batch azure-storage-blob jsonschema pyyaml
```
Also note that this module has only been tested with Python 3.7

### Create the Required Azure resources

Using Azure batch requires an azure account, and we'll demonstrate how to run
this module using the [azure command line tool]().

After logging into the console with `az login`,  you'll need to create an azure
resource group into which the batch account is created.  In addition, the 
azure batch service requires a storage account which is used to keep track of
details of the jobs and tasks.

Although the resource group, storage account and batch account could have
different names, for sake of exposition, we'll give them all the same name and
locate them in the US West 2 region, like so:

###### Powershell
```ps1
# parameters
$name = "sparsesctest"
$location = "westus2"
# create the resources
az group create -l $location -n $name
az storage account create -n $name -g $name
az batch account create -l $location -n $name -g $name --storage-account $name
```

###### Bash
```bash
# parameters
set name=sparsesctest
set location=westus2
# create the resources
az group create -l %location% -n %name%
az storage account create -n %name% -g %name%
az batch account create -l %location% -n %name% -g %name% --storage-account %name%
```
###### CMD
```bash
# parameters
name="sparsesctest"
location="westus2"
# create the resources
az group create -l $location -n $name
az storage account create -n $name -g $name
az batch account create -l $location -n $name -g $name --storage-account $name
```

*(Aside: since we're using the `name` for parameter for the resource group
storage account and batch account, it must consist of 3-24 lower case
letters and be unique across all of azure)* 

### Gather Resource Credentials

We'll need some information about the batch and storage accounts in order
to create and run batch jobs. We can create bash variables that contain the
information that the SparseSC azure batch client will require, with the
following:

###### Powershell
```ps1
$BATCH_ACCOUNT_NAME = $name
$BATCH_ACCOUNT_KEY =  az batch account keys list -n $name -g $name --query primary
$BATCH_ACCOUNT_URL = "https://$name.$location.batch.azure.com"
$STORAGE_ACCOUNT_NAME = $name
$STORAGE_ACCOUNT_KEY = az storage account keys list -n $name --query [0].value
```
###### Bash
```bash
export BATCH_ACCOUNT_NAME=$name
export BATCH_ACCOUNT_KEY=$(az batch account keys list -n $name -g $name --query primary)
export BATCH_ACCOUNT_URL="https://$name.$location.batch.azure.com"
export STORAGE_ACCOUNT_NAME=$name
export STORAGE_ACCOUNT_KEY=$(az storage account keys list -n $name --query [0].value)
```
###### CMD
```bash
set BATCH_ACCOUNT_NAME=%name%
set STORAGE_ACCOUNT_NAME=%name%
set BATCH_ACCOUNT_URL=https://%name%.%location%.batch.azure.com
for /f %i in ('az batch account keys list -n %name% -g %name% --query primary') do set BATCH_ACCOUNT_KEY=%i
for /f %i in ('az storage account keys list -n %name% --query [0].value') do set STORAGE_ACCOUNT_KEY=%i
```

We could of course echo these to the console and copy/paste the values into the
BatchConfig object below. However we don't need to do that if we run python
from within the same environment (terminal session), as these environment
variables will be used by the `azure_batch_client` if they are not provided
explicitly.

## Prepare parameters for the Batch Job

Parameters for a batch job can be created using `fit()` by providing a directory where the batch parameters should be stored:
```python
from SparseSC import fit
batch_dir = os.path.expanduser("/path/to/my/batch/data/")

# initialize the batch parameters in the directory `batch_dir`
fit(x, y, ... , batchdir = batch_dir)
```

## Executing the Batch Job

In the following Python script, a Batch configuration is created and the batch
job is executed with Azure Batch. Note that the Batch Account and Storage
Account details can be provided directly to the BatchConfig, with default
values taken from the system environment.

```python
import os
from datetime import datetime
from SparseSC.utils.AzureBatch import BatchConfig, run as run_batch_job, aggregate_batch_results

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
	# Keep the pool around after the run, which saves time when doing
	# multiple batch jobs, as it typically takes a few minutes to spin up a
	# pool of VMs
    DELETE_POOL_WHEN_DONE=False,
	# Keeping the job details can be useful for debugging:
    DELETE_JOB_WHEN_DONE=False
    )

# run the batch job
run_batch_job(my_config)

# aggregate the results into a fitted model instance
fitted_model = aggregate_batch_results(batchdir)
```

Note that the pool configuration will only be used to create a new pool if no pool 
by the id `POOL_ID` exists.  If the pool already exists, these parameters are
ignored and will *not* update the pool configuration.  Changing pool attrributes 
such as type or quantity of nodes can be done through the [Azure Portal](https://portal.azure.com/), [Azure Batch Explorer](https://azure.github.io/BatchExplorer/) or any of the APIs.

## Cleaning Up

In order to prevent unexpected charges, the resource group, including all the
resources it contains, such as the storge account and batch pools, can be
removed with the following command.

###### Powershell and Bash
```ps1
az group delete -n $name
```
###### CMD
```bat
az group delete -n %name%
```
```eval_rst
.. autoclass:: SparseSC.utils.AzureBatch.BatchConfig
    :members:
    :show-inheritance:

```

```eval_rst
.. autoclass:: SparseSC.utils.AzureBatch.run
    :members:
    :show-inheritance:

```

