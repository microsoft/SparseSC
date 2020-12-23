#On Windows can use 'nmake'
#  Incl w/ VS 2015 or VS 2017 (w/ "Desktop development with C++" components)
#  Incl in path or use the "Developer Command Prompt for VS...."
#  Can run 'nmake /NOLOGO ...' to remove logo output.
#nmake can't do automatic (pattern) rules like make (has inference rules which aren't cross-platform)

# For linux, can use Anaconda or if using virtualenv, install virtualenvwrapper 
# and alias activate->workon.
# TO DO: make activate read the env variables.

help:
	@echo "Use one of the common targets: pylint, package, readmedocs, htmldocs"

#Allow for slightly different commands for nmake and make
#NB: Don't always need the different DIR_SEP
# \
!ifndef 0 # \
# nmake specific code here \
RMDIR_CMD = rmdir /S /Q # \
RM_CMD = del # \
DIR_SEP = \ # \
!else
# make specific code here
RMDIR_CMD = rm -rf
RM_CMD = rm
DIR_SEP = /# \
# \
!endif

#Creates a "Source Distribution" and a "Pure Python Wheel" (which is a bit easier for user)
package: package_both

package_both:
	python setup.py sdist bdist_wheel

package_sdist:
	python setup.py sdist

package_bdist_wheel:
	python setup.py bdist_wheel
	
pypi_upload:
	twine upload dist/*
#python -m twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

pylint:
	-mkdir build
	-pylint SparseSC > build$(DIR_SEP)pylint_msgs.txt

SOURCEDIR     = docs/
BUILDDIR      = docs/build
BUILDDIRHTML  = docs$(DIR_SEP)build$(DIR_SEP)html
BUILDAPIDOCDIR= docs$(DIR_SEP)build$(DIR_SEP)apidoc

htmldocs:
	-$(RMDIR_CMD) $(BUILDDIRHTML)
	-$(RMDIR_CMD) $(BUILDAPIDOCDIR)
#	sphinx-apidoc -f -o $(BUILDAPIDOCDIR)/SparseSC SparseSC
#	$(RM_CMD) $(BUILDAPIDOCDIR)$(DIR_SEP)SparseSC$(DIR_SEP)modules.rst
	@python -msphinx -b html -T -E "$(SOURCEDIR)" "$(BUILDDIR)" $(O)

examples:
	python example-code.py
	python examples/fit_poc.py

tests:
	python -m unittest test.test_fit.TestFitForErrors test.test_fit.TestFitFastForErrors test.test_normal.TestNormalForErrors test.test_estimation.TestEstimationForErrors

#tests_both:
#	activate SparseSC_36 && python -m unittest test.test_fit

#add examples here when working
check: pylint package_bdist_wheel tests_both

#Have to strip because of unfixed https://github.com/jupyter/nbconvert/issues/503
examples/DifferentialTrends.py: examples/DifferentialTrends.ipynb
	jupyter nbconvert examples/DifferentialTrends.ipynb --to script
	cd examples && python strip_magic.py

examples/DifferentialTrends.html: examples/DifferentialTrends.ipynb
	jupyter nbconvert examples/DifferentialTrends.ipynb --to html

examples/DifferentialTrends.pdf: examples/DifferentialTrends.ipynb
	cd examples && jupyter nbconvert DifferentialTrends.ipynb --to pdf

clear_ipynb_output:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace examples/DifferentialTrends.ipynb
gen_ipynb_output:
	jupyter nbconvert --to notebook --execute examples/DifferentialTrends.ipynb

#Have to cd into subfulder otherwise will pick up potential SparseSC pkg in build/
#TODO: Make the prefix filter automatic
#TODO: check if this way of doing phony targets for nmake works with make
test/SparseSC_36.yml: .phony
	activate SparseSC_36 && cd test && conda env export > SparseSC_36.yml
	echo Make sure to remove the last prefix line and the pip sparsesc line, as user does pip install -e for that
.phony:

#Old:
# Don't generate requirements-rtd.txt from conda environments (e.g. pip freeze > rtd-requirements.txt)
# 1) Can be finicky to get working since using pip and docker images and don't need lots of packages (e.g. for Jupyter)
# 2) Github compliains about requests<=2.19.1. Conda can't install 2.20 w/ Python <3.6. Our env is 3.5, but RTD uses Python3.7
# Could switch to using conda 
#doc/rtd-requirements.txt:

conda_env_upate:
	deactivate && conda env update -f test/SparseSC_36.yml

#Just needs to be done once
conda_env_create:
	conda env create -f test/SparseSC_36.yml

jupyter_DifferentialTrends:
	START jupyter notebook examples/DifferentialTrends.ipynb > jupyter_launch.log
  