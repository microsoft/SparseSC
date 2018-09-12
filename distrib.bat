@rem Build the Sparse Synthetic Control package
@echo off

setlocal
set outdir=%1
if "%1"=="" set outdir=dist


echo ##############################
echo Building distribution
echo ##############################
@rem Creates a "Source Distribution" and a "Pure Python Wheel" (which is a bit easier for user)
python setup.py sdist -d %outdir%
python setup.py bdist_wheel -d %outdir%
