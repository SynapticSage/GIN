"""
Script that focuses on iterating through all of the odors
and running a prediction script
"""

import gin
import argparse
import os

gin_path = gin.__path__

parser = argparse.ArgumentParser()
parser.add_argument("--archive", desc="which archive to iterate",
                    default="leffingwell")
parser.add_argument("--script", desc="which script to run",
                    default=os.path.join(gin_path, "..", "script",
                                  "Pyrfume_RF_GNN_singleOdor.py"))
args = parser.parse_args()

odors = gin.data.pyr.list_descriptors(args.archive)

for odor in odors:

  # build call
  command = ["python",
    args.script,
    "--archive",
    args.archive,
    "--descriptor",
    odor]
    
  # and now let's call it

  print("Running cmd = ", " ".join(cmd))
  out = os.popen(cmd)
  print("Output", "----", out.readlines(), sep="\n")
