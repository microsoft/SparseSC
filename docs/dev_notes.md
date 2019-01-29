Building docs
=============
Required python packages: `sphinx`, `recommonmark`, `sphinx-markdown-tables`
Index HTML file is at `docs/build/html/index.html`

Testing
=======
We use the built-in `unittest`. Can run from makefile using the `tests` target or you can run python directly from the repo root using the following types of commands:

```python
python -m unittest test/test_fit.py #module
python -m unittest test.test_fit.TestFit #class
python -m unittest test.test_fit.TestFit.test_retrospective #function
```

Release Process
===============
* Make sure tests run clean
* Check that tests run clean
* If new version, check that it's been updated in `SparseSC/__init__.py`
* Updated `Changelog.md`
* Tag/Release in version control