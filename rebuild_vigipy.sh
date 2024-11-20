#!/bin/bash 

python -m pip uninstall vigipy 
python setup.py bdist_wheel
python -m pip install ./dist/vigipy-1.4.0-py3-none-any.whl