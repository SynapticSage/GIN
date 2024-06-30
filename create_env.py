#!/bin/bash

python3.12 -m venv env
source env/bin/activate
pip install -e .
echo "python location: `which python`"
pip install ipython
