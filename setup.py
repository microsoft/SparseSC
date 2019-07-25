#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import re
from glob import glob
from os.path import basename, dirname, join, splitext, abspath
from setuptools import find_packages, setup
import codecs

# Allow single version in source file to be used here
# From https://packaging.python.org/guides/single-sourcing-package-version/
def read(*parts):
    # intentionally *not* adding an encoding option to open
    # see here: https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    here = abspath(dirname(__file__))
    return codecs.open(join(here, *parts), "r").read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="SparseSC",
    version=find_version("src", "SparseSC", "__init__.py"),
    description="Sparse Synthetic Controls",
    license="MIT",
    long_description="%s\n%s"
    % (
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.md")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.md")),
    ),
    long_description_content_type="text/markdown",
    author="Microsoft Research",
    url="https://github.com/Microsoft/SparseSyntheticControls",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
    ],
    keywords=["Sparse", "Synthetic", "Controls"],
    install_requires=["numpy", "Scipy", "scikit-learn"],
    entry_points={
        "console_scripts": [
            "scgrad=SparseSC.cli.scgrad:main",
            "daemon=SparseSC.cli.daemon_process:main",
            "stt=SparseSC.cli.stt:main",
        ]
    },
)
