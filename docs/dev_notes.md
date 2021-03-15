# Developer Notes 

## Python environments

You can create Anaconda environments using
```bash
conda env create -f test/SparseSC_36.yml
```
You can can do `update` rather than `create` to update existing ones (to avoid [potential bugs](https://stackoverflow.com/a/46114295/3429373) make sure the env isn't currently active).

Note: When regenerating these files (`conda env export > test/SparseSC_*.yml`) make sure to remove the final `prefix` line since that's computer specific. You can do this automatically on Linux by inserting `| grep -v "prefix"` and on Windows by inserting `| findstr -v "prefix"`.

## Building the docs
Requires Python >=3.6 and packages: `sphinx`, `recommonmark`, `sphinx-markdown-tables`. 
Use `(n)make htmldocs` and an index HTML file is madeat `docs/build/html/index.html`.

To build a mini-RTD environment to test building docs:
1) You can make a new environment with Python 3.7 (`conda create -n SparseSC_37_rtd python=3.7`)
2) update `pip` (likely fine).
3) `pip install --upgrade --no-cache-dir -r docs/rtd-base.txt` . This file is loosely kept in sync by looking at the install commands on the rtd run.
4) `pip install --exists-action=w --no-cache-dir -r docs/rtd-requirements.txt` . This file doesn't list the full environment versions because that causes headaches when the rtd base environment got updated. It downgrades Sphinx to a known good version that allows markdown files to have math in code quotes ([GH Issues](https://github.com/readthedocs/recommonmark/issues/133)) (there might be higher ones that also work, didn't try). 

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

