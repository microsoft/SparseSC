#On Windows can use 'nmake'
#  Incl w/ VS 2015 or VS 2017 (w/ "Desktop development with C++" components)
#  Incl in path or use the "Developer Command Prompt for VS...."

help:
	@echo "Use one of the common targets: pylint, package, docs"

#Creates a "Source Distribution" and a "Pure Python Wheel" (which is a bit easier for user)
package:
	python setup.py sdist -d dist
	python setup.py bdist_wheel -d dist

readmedocs:
	pandoc README.md -f markdown -t latex -o docs/SyntheticControlsReadme.pdf
	pandoc README.md -f markdown -t docx -o docs/SyntheticControlsReadme.docx

pylint:
	pylint RidgeSC

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = python -msphinx
SPHINXPROJ    = RidgeSC
SOURCEDIR     = docs/
BUILDDIR      = docs/build
BUILDDIRHTML  = docs\build\html
BUILDAPIDOCDIR= docs\build\apidoc
SPHINXAPIDOC  = sphinx-apidoc
htmldocs:
	-rmdir /S /Q $(BUILDDIRHTML)
	$(SPHINXAPIDOC) -f -o $(BUILDAPIDOCDIR)/RidgeSC RidgeSC
  del $(BUILDAPIDOCDIR)\modules.rst
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
