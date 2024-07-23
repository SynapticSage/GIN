#!/bin/bash
script_folder=$(dirname $(realpath $0))
py_file=${2:-"./Pyrfume_Floral.py"}
output_file="${py_file%.py}.ipynb"
echo "Converting $1 to jupyter notebook at $script_folder"

# Corrected the assignment of jupytext path
jupytext=$(which jupytext)

# Ensure the jupytext command is available
if [ -z "$jupytext" ]; then
  echo "jupytext not found. Please ensure it is installed and in your PATH."
  exit 1
fi

# Convert the Python script to a Jupyter notebook
$jupytext --to notebook $py_file

echo "Conversion complete. Jupyter notebook saved in $script_folder"

# Move to appropriate folder
echo cp ../notebook/$output_file ../notebook/${output_file}.bak.$(date +%Y%m%d%H%M%S)
cp ../notebook/$output_file ../notebook/${output_file}.bak.$(date +%Y%m%d%H%M%S)
mv $output_file ../notebook/
