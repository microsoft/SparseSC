# Batch config testing

## Install Batch Config

```shell
pip install git+https://github.com/jdthorpe/batch-configo
```

## Gather required credentials

These commands assume the CMD terminal. For other terminals, [see here](https://jdthorpe.github.io/super-batch-docs/create-resources).

```bat
az login
# optionally: az account set -s xxxxxxx-xxxx-xxxx-xxxx-xxxxxxxx

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
```

## run the tests

```bat
cd test\AzureBatch
rm -rf data
python test_batch_build.py
python test_batch_run.py
python test_batch_aggregate.py
```
