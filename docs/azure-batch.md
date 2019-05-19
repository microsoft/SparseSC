
## Setup

The azure batch client requires some additional dependencies which can be installed via:

```bash
pip install  azure-batch azure-storage-blob jsonschema pyyaml
```

   ```
   az something something ...
   ```

pip install git+https://github.com/Microsoft/sparsesc.git@x-grad-daemon


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

We could of course echo these to the console to get their value, however we
don't need to do that if we run python from within the same session, as these
variales will be found by the SparseSC batch client if they are not provided
explicitly.

# Using the Sparse SC with Azure Batch

```python
from datetime import datetime
from SparseSC.utils.azure_batch_client import BatchConfig


timestamp = datetime.utcnow().strftime("%H%M%S")

BatchConfig


    POOL_ID=" test_pool",
    POOL_LOW_PRIORITY_NODE_COUNT=5,
    POOL_VM_SIZE: str
    JOB_ID: str
    CONTAINER_NAME: str
    BATCH_DIRECTORY: str





```

# Cleaning Up

In order to prevent unexpected charges, the resource group, including all the
resources it contains, such as the storge account and batch pools, can be
removed with the following command.

```bash
az group delete -n $name
```
