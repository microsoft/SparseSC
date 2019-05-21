# Azure Batch

Fitting a Sparse Synthetic Controls model can result in a very long running
time. Fortunately much of the work can be done in parallel and executed in
the cloud, and the SparseSC package comes with an Azure Batch utility which
can be used to fit a Synthetic controls model using Azure Batch.  There are
code examples for [Windows CMD](https://sparsesc.readthedocs.io/en/latest/azure-batch-cmd.html),
[Bash](https://sparsesc.readthedocs.io/en/latest/azure-batch-bash.html) and ~~Powershell~~ (soon).


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

