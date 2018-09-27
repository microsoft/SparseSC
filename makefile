#On Windows can use 'nmake'
#  Incl w/ VS 2015 or VS 2017 (w/ "Desktop development with C++" components)
#  Incl in path or use the "Developer Command Prompt for VS...."

#Creates a "Source Distribution" and a "Pure Python Wheel" (which is a bit easier for user)
package:
	python setup.py sdist -d dist
	python setup.py bdist_wheel -d dist

docs:
	pandoc README.md -f markdown -t latex -o ../SyntheticControlsReadme.pdf
	pandoc README.md -f markdown -t docx -o ../SyntheticControlsReadme.docx

pylint:
	pylint RidgeSC
