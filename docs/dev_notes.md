# Developer Notes 

## Python environments

You can create Anaconda environments using
```bash
conda env create -f test/SparseSC_36.yml
```
You can can do `update` rather than `create` to update existing ones (to avoid [potential bugs](https://stackoverflow.com/a/46114295/3429373) make sure the env isn't currently active).

Note: When regenerating these files (`conda env export > test/SparseSC_*.yml`) make sure to remove the final `prefix` line since that's computer specific.

## Building the docs
Requires Python >=3.6 and packages: `sphinx`, `recommonmark`, `sphinx-markdown-tables`. 
Use `(n)make htmldocs` and an index HTML file is madeat `docs/build/html/index.html`.

To build a mini-RTD environment to test building docs:
1) You can make a new environment with Python 3.7
2) update `pip`
3) `pip install --upgrade --no-cache-dir -r docs/rtd_base.txt`
4) `pip install --exists-action=w --no-cache-dir -r docs/rtd-requirements.txt`

There are some errors from our setup that aren't present in RTD (they use Python 3.7 and a pip environment with the latest packages).

## Running examples
The Jupyter notebooks require `matplotlib`, `jupyter`, and `notebook`.

## Testing
We use the built-in `unittest`. Can run from makefile using the `tests` target or you can run python directly from the repo root using the following types of commands:

```bash
python -m unittest test/test_fit.py #file (only Python >=3.5)
python -m unittest test.test_fit #module
python -m unittest test.test_fit.TestFit #class
python -m unittest test.test_fit.TestFit.test_retrospective #function
```


## Release Process
* Ensure the makefile target `check`  (which does pylint, tests, doc building, and packaging) runs clean
* If new version, check that it's been updated in `SparseSC/src/__init__.py`
* Updated `Changelog.md`
* Tag/Release in version control

