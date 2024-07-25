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
  for i in range(10):
    filepath = path.strip('/') + '/' + f"nested_cross_validation_score_{i}.csv"
    if not os.path.exists(filepath):
      raise ValueError(f"File does not exist: {filepath}")
    dfs.append(pd.read_csv(filepath))

  # Compute average score  
  scores = [df['0'][0] for df in dfs]
  avg = sum(scores) / len(scores)

  print(f"Final scores: {scores}")
  print(f"Average score: {avg}")
  
  # Save as a text file in provided directory
  file = open(path.strip('/') + '/' + "average_accuracy.txt", 'w')
  file.write(str(avg))
  file.close()
