# Running Jobs in Parallel with Azure Batch

Fitting a Sparse Synthetic Controls model can result in a very long running
time. Fortunately much of the work can be done in parallel and executed in
the cloud, and the SparseSC package comes with an Azure Batch utility which
can be used to fit a Synthetic controls model using Azure Batch.  There are
code examples for Windows CMD, Bash and Powershell.

## Setup

Running SparseSC with Azure Batch requires the `super_batch` library which
can be installed with:

```bash
pip install git+https://github.com/jdthorpe/batch-config.git
```

Also note that this module has only been tested with Python 3.7

### Create the Required Azure resources

Running SparseSC with Azure Batch requires a an Azure account and handful of
resources and credentials. These can be set up by following along with
[section 4 of the super-batch README](https://github.com/jdthorpe/batch-config#step-4-create-the-required-azure-resources).

### Prepare parameters for the Batch Job

The parameters required to run a batch job can be created using `fit()` by
providing a directory where the parameters files should be stored:

```python
from SparseSC import fit
batch_dir = "/path/to/my/batch/data/"

# initialize the batch parameters in the directory `batch_dir`
fit(x, y, ... , batchDir = batch_dir)
```

### Executing the Batch Job

In the following Python script, a Batch configuration is created and the
batch job is executed with Azure Batch. Note that in the following script,
the various Batch Account and Storage Account credentials are taken from
system environment varables, as in the [super-batch readme](https://github.com/jdthorpe/batch-config#step-4-create-the-required-azure-resources).

```python
import os
from datetime import datetime
from super_batch import Client
from SparseSC.utils.AzureBatch import (
    DOCKER_IMAGE_NAME,
    create_job,
)
# Batch job names must be unique, and a timestamp is one way to keep it uniquie across runs
timestamp = datetime.utcnow().strftime("%H%M%S")
batch_dir = "/path/to/my/batch/data/"

batch_client = Client(
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
    BATCH_DIRECTORY= batch_dir,
    # Keep the pool around after the run, which saves time when doing
    # multiple batch jobs, as it typically takes a few minutes to spin up a
    # pool of VMs. (Optional. Default = False)
    DELETE_POOL_WHEN_DONE=False,
    # Keeping the job details can be useful for debugging:
    # (Optional. Default = False)
    DELETE_JOB_WHEN_DONE=False
)

create_job(batch_client, batch_dir)
# run the batch job
batch_client.run()

# aggregate the results into a fitted model instance
fitted_model = aggregate_batch_results(batch_dir)
```

## Cleaning Up

When you are done fitting your model with Azure Batch be sure to 
[clean up your Azure Resources](https://github.com/jdthorpe/batch-config#step-6-clean-up)
in order to prevent unexpected charges on your Azure account.

## Solving

The Azure batch will just vary one of the penalty parameters. You should therefore not specify the
simplex constraint for the V matrix as then it will be missing one degree of freedom.

## FAQ

1. What if I get disconnected while the batch job is running?

    Once the pool and the job are created, they will keep running until the
    job completes, or your delete the resources. You can reconnect create the
    `batch_client` as in the example above and then reconnect to the job and
    download the results with:

    ```python
    batch_client.load_results()
    fitted_model = aggregate_batch_results(batch_dir)
    ```

    In fact, if you'd rather not wait for the job to compelte, you can
    add the parameter `batch_client.run(... ,wait=False)` and the
    `run_batch_job` will return as soon as the job and pool configuration
    have been createdn in Azure.

1. `batch_client.run()` or `batch_client.load_results()` complain that the
    results are in complete. What happened?

   Typically this means that one or more of the jobs failed, and a common
   reason for the job to fail is that the VM runs out of memory while
   running the batch job.  Failed Jobs can be viewed in either the Azure
   Batch Explorer or the Azure Portal. The `POOL_VM_SIZE` use above
   ("STANDARD_A1_v2") is one of the smallest (and cheapest) VMs available
   on Azure.  Upgrading to a VM with more memory can help in this
   situation.

1. Why does `aggregate_batch_results()` take so long?

   Each batch job runs a single gradient descent in V space using a subset
   (Cross Validation fold) of the data and with a single pair of penalty
   parameters, and return the out of sample error for the held out samples.
   `aggregate_batch_results()` very quickly aggregates these out of sample
   errors and chooses the optimal penalty parameters given the `choice`
   parameter provided to `fit()` or `aggregate_batch_results()`.  Finally,
   with the selected parameters, a final gradient descent is run using the
   full dataset which will be larger than the and take longer as the rate
   limiting step
   ( [scipy.linalg.solve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve.html) )
   has a running time of
   [`O(N^3)`](https://stackoverflow.com/a/12665483/1519199). While it is
   possible to run this step in parallel as well, it hasn't yet been
   implemented.
