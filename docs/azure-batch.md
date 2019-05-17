
## Setup

1. The azure batch client requires some additional dependencies which can be installed via:

	```bash
	pip install  azure-batch azure-storage-blob jsonschema pyyaml
	```

1. Using Azure batch requires an azure account which can be created
   [here](), and we'll demonstrate how to run this module using the [azure
   command line tool]() which will also need to be installed.


1. After logging into the console with `az login`,  we can create an azure
   Resource Group which will help to keep track of billing charges and also
   help to clean up resourses later on, to prevent incurring additional
   fees when we're done with these resources

   ```
   az group create ...
   ```

1. Next we can create an azure batch resource along with the storage
   account that azure batch uses to keep track of our tasks, like so:
   ```
   az something something ...
   ```
   and with that we're ready to begin using azure batch. 

1. Finally, we'll need some information about our storage and batch
   accounts that will allow the batch client to run jobs in azure batch. 
   ```
   az something something ...
   ```

pip install git+https://github.com/Microsoft/sparsesc.git@x-grad-daemon

