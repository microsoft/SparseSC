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
package: package_bdist_wheel

package_sdist:
	python setup.py sdist -d dist

package_bdist_wheel:
	python setup.py bdist_wheel -d dist

readmedocs:
	pandoc README.md -f markdown -t latex -o docs/SyntheticControlsReadme.pdf
	pandoc README.md -f markdown -t docx -o docs/SyntheticControlsReadme.docx

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
	@python -msphinx -b html -q "$(SOURCEDIR)" "$(BUILDDIR)" $(O)

examples:
	python example-code.py
	python examples/fit_poc.py

#Python 2.7 can do 'test.test_fit' but not 'test/test_fit.py'
tests:
	python -m unittest test.test_fit

tests_both:
	activate SparseSC_27 && python -m unittest test.test_fit
	activate SparseSC_35 && python -m unittest test.test_fit

#add examples here when working
check: pylint package_bdist_wheel tests_both

examples/DifferentialTrends.py: examples/DifferentialTrends.ipynb
	jupyter nbconvert examples/DifferentialTrends.ipynb --to script

examples/DifferentialTrends.html: examples/DifferentialTrends.ipynb
	jupyter nbconvert examples/DifferentialTrends.ipynb --to html

examples/DifferentialTrends.pdf: examples/DifferentialTrends.ipynb
	cd examples && jupyter nbconvert DifferentialTrends.ipynb --to pdf

#Have to cd into subfulder otherwise will pick up potential SparseSC pkg in build/
#TODO: Make the prefix filter automatic
#TODO: check if this way of doing phony targets for nmake works with make
test/SparseSC_27.yml: .phony
	activate SparseSC_27 && cd test && conda env export > SparseSC_27.yml
	echo Make sure to remove the last prefix line
test/SparseSC_35.yml: .phony
	activate SparseSC_35 && cd test && conda env export > SparseSC_35.yml
	echo Make sure to remove the last prefix line
.phony:

#TODO: Github compliains about requests<=2.19.1. Conda can't install 2.20 w/ Python <3.6.
#doc/rtd-requirements.txt:
#	activate SparseSC_35 && cd doc && pip freeze > rtd-requirements.txt

conda_env_upate:
	deactivate && conda env update -f test/SparseSC_27.yml
	deactivate && conda env update -f test/SparseSC_35.yml

#Just needs to be done once
conda_env_create:
	conda env create -f test/SparseSC_27.yml
	conda env create -f test/SparseSC_35.yml
