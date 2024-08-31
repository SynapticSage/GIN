"""
Script that focuses on iterating through all of the odors
and running a prediction script
"""

import gin
import argparse
import os

gin_path = gin.__path__

parser = argparse.ArgumentParser()
parser.add_argument("--archive", 
                    default="leffingwell",
                    type=str,
                    help="which archive to iterate",
                    )
parser.add_argument("--script", 
                    help="which script to run",
                    type=str,
                    default=os.path.join(*gin_path, "..",
                                         "scripts",
                                         "Pyrfume_RF_GNN_singleOdor.py")
                    )
args = parser.parse_args()
args.script = os.path.abspath(args.script)


odors = gin.data.pyr.list_descriptors(args.archive)

for odor in odors[1:]:

  # build call
  command = ["python",
    args.script,
    "--archive",
    args.archive,
    "--descriptor",
    odor]
    
  # and now let's call it

  print("Running cmd = ", " ".join(command))
  out = os.popen(" ".join(command))
  print("Output", "----", out.read(), sep="\n")
