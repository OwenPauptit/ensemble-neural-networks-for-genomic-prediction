import os
import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
  # Validate command line arguments
  args = sys.argv[1:]
  if len(args) != 1:
    raise ValueError("This script requires 1 positional argument, the location of the subdirectory with the output files present")
  path = args[0]
  if not os.path.exists(path):
    raise ValueErorr("Provided path does not exist")

  dfs = []
  # Extract data from output files in directory
  for i in range(40):
    filepath = path.strip('/') + '/' + f"env_score_{i}"
    if not os.path.exists(filepath):
      break
    dfs.append(pd.read_csv(filepath))

  dfs = pd.concat(dfs)
  dfs.to_csv(path.strip('/') + '/' + "combined_scores.csv")

  print("Combined dataframe: ")
  print(dfs)
  
 
