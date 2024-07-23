#!/bin/bash
script_folder=$(dirname $(realpath $0))
notebook_file=${2:-"./Pyrfume_Floral.ipynb"}
output_file="${notebook_file%.ipynb}.py"

# Corrected the assignment of jupytext path
jupytext=$(which jupytext)

# Ensure the jupytext command is available
if [ -z "$jupytext" ]; then
  echo "jupytext not found. Please ensure it is installed and in your PATH."
  exit 1
fi

# Convert the Jupyter notebook to a Python script
$jupytext --to py "$notebook_file"

# Check if the conversion was successful
if [ -f "$output_file" ]; then
  echo "Conversion complete. Python script saved as $output_file"
else
  echo "Conversion failed. Please check the notebook file and try again."
fi

cp ../scripts/$output_file ../scripts/${output_file}.bak.$(date +%Y%m%d%H%M%S)
mv $output_file ../scripts/
