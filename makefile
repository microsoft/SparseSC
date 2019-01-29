#On Windows can use 'nmake'
#  Incl w/ VS 2015 or VS 2017 (w/ "Desktop development with C++" components)
#  Incl in path or use the "Developer Command Prompt for VS...."

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
package:
	python setup.py sdist -d dist
	python setup.py bdist_wheel -d dist

readmedocs:
	pandoc README.md -f markdown -t latex -o docs/SyntheticControlsReadme.pdf
	pandoc README.md -f markdown -t docx -o docs/SyntheticControlsReadme.docx

pylint:
	-mkdir build
	pylint SparseSC > build$(DIR_SEP)pylint_msgs.txt

SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = SparseSC
SOURCEDIR     = docs/
BUILDDIR      = docs/build
SPHINXAPIDOC  = sphinx-apidoc
BUILDDIRHTML  = docs$(DIR_SEP)build$(DIR_SEP)html
BUILDAPIDOCDIR= docs$(DIR_SEP)build$(DIR_SEP)apidoc

htmldocs:
	-$(RMDIR_CMD) $(BUILDDIRHTML)
	-$(RMDIR_CMD) $(BUILDAPIDOCDIR)
	$(SPHINXAPIDOC) -f -o $(BUILDAPIDOCDIR)/SparseSC SparseSC
	$(RM_CMD) $(BUILDAPIDOCDIR)$(DIR_SEP)SparseSC$(DIR_SEP)modules.rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
