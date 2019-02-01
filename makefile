#On Windows can use 'nmake'
#  Incl w/ VS 2015 or VS 2017 (w/ "Desktop development with C++" components)
#  Incl in path or use the "Developer Command Prompt for VS...."
#  Can run 'nmake /NOLOGO ...' to remove logo output.

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
DIR_SEP = / # \
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
	@python -msphinx -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(O)

examples:
	python example-code.py
	python examples/fit_poc.py

#Python 2.7 can do 'test.test_fit' but not 'test/test_fit.py'
tests:
	python -m unittest test.test_fit

tests_both:
	activate SparseSC_27 && python -m unittest test.test_fit
	activate SparseSC_35 && python -m unittest test.test_fit

check: pylint package_bdist_wheel tests_both examples
