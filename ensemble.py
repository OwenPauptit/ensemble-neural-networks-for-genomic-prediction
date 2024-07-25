import os, sys
import pandas as pd
import io
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from multiprocessing import Pool
import random

import warnings
warnings.simplefilter('ignore')

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

print("Imported modules successfully")

# Keeping note of which columns are which
columns_to_drop = ['1','2','3']
yield_col = '3'

class CDBNDataset(torch.utils.data.Dataset):
  '''
  Prepare the CDBN dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
    if not torch.is_tensor(X) and not torch.is_tensor(y):
      # Apply scaling if necessary
      if scale_data:
          X = StandardScaler().fit_transform(X)
      self.X = torch.from_numpy(X)
      self.y = torch.from_numpy(y)

  def __len__(self):
      return len(self.X)

  def __getitem__(self, i):
      return self.X[i], self.y[i]

class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, X, y, input_dim, hidden_dims=[5], lr=0.1, batch_size=16, output_dim=1):
    super().__init__()
    self.lr = lr
    self.batch_size = batch_size

    layer_list = self.__constructLayerList(input_dim, hidden_dims, output_dim)
    self.layers = nn.Sequential(*layer_list)

    X = np.array(X)
    y = np.array(y)
    dataset = CDBNDataset(X, y)
    self.trainloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    self.loss_fn = nn.MSELoss()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

  def trainOneEpoch(self):
    # Perform backward propogation one batch at a time
    for i, data_batch in enumerate(self.trainloader, 0):

      # Get and prepare inputs
      inputs, targets = data_batch
      inputs, targets = inputs.float(), targets.float()
      targets = targets.reshape((targets.shape[0], 1))

      # Reset optimiser
      self.optimizer.zero_grad()

      # Forward pass
      outputs = self(inputs)
      loss = self.loss_fn(outputs, targets)

      # Backward pass
      loss.backward()
      self.optimizer.step()

  def __constructLayerList(self, input_dim, hidden_dims, output_dim):
    '''
    Helper method to construct list of MLP layers during initialisation
    '''
    layer_list = [
        nn.Linear(input_dim, hidden_dims[0]),
        nn.ReLU()
    ]
    for i in range(len(hidden_dims) - 1):
      layer_list.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
      layer_list.append(nn.ReLU())
    layer_list.append(nn.Linear(hidden_dims[-1], output_dim))
    return layer_list

  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)

  def predict(self, X0):
    '''predict values from matrix X0'''
    X0 = torch.from_numpy(np.array(X0)).float()
    return self(X0)

class Ensemble:
 '''An ensemble of neural networks'''
  def __init__(self, training_data, method="l1o", lr=0.1, batch_size=16, hidden_dims=[5], num_epochs=50, compute_loss=True, n_weak_learners=21):
    '''
    training_data - dataframe in CDBN format
    method - ensemble method to use, choose from l1o, bagging, mlp, 1by1
    lr - learning rate of each NN, between 0 and 1
    batch_size - batch_size to use during training, positive integer
    hidden_dims - width of each hidden layer as a list
    num_epochs - cut off for maximum number of epochs
    compute_loss - whether or not to compute the loss during training - strongly advised
    n_weak_learners - (only for bagging) the number of weak learners in the ensemble
    '''
    self.data = training_data
    self.method = method
    self.lr = lr
    self.batch_size = batch_size
    self.hidden_dims = hidden_dims
    self.num_epochs = num_epochs
    self.models = []
    self.subsets = []
    self.loss = []
    self.unique_groups = self.data['1'].unique()
    self.compute_loss = compute_loss

    # Only used for bagging
    self.n_weak_learners =  n_weak_learners
    self.n_groups_per_learner = len(self.unique_groups)

    if self.compute_loss:
      self.response = np.array(self.data[yield_col])
      self.matrix = self.data.drop(columns_to_drop, axis=1)

    self.__initialiseModels()

  def __initialiseModels(self):
    '''helper method for constructor'''
    if self.method == "bagging":
      self.__bagging()
    elif self.method == "l1o":
      self.__leaveOneOut()
    elif self.method == "mlp":
      self.__oneModel()
    elif self.method == "1by1":
      self.__1by1()
    else:
      raise ValueError("Invalid method")

    for i in range(len(self.subsets)):
      X = self.subsets[i].drop(columns_to_drop, axis = 1)
      y = self.subsets[i][yield_col]
      mlp = MLP(X, y, input_dim = X.shape[1], hidden_dims = self.hidden_dims, output_dim = 1, lr = self.lr, batch_size = self.batch_size)
      self.models.append(mlp)

  def __str__(self):
    print(f"Ensemble of {len(self.models)} models using {self.method} method")

  def __bagging(self):
    '''
    Create a bootstrap aggregation ensemble.
    '''
    for i in range(self.n_weak_learners):
      # Randomly sample locations with replacement
      bootstrapped_groups = np.random.choice(self.unique_groups, size = self.n_groups_per_learner, replace = True)
      sample = pd.DataFrame(columns = self.data.columns)

      for group in bootstrapped_groups:
        sample = pd.concat([sample, self.data[self.data['1'] == group]])

      self.subsets.append(sample)

  def __leaveOneOut(self):
    '''
    Create a leave-one-out ensemble.
    '''
    for env in self.unique_groups:
      without_env = self.data[self.data['1'] != env]
      self.subsets.append(without_env)

  def __oneModel(self):
    '''
    Create a single neural network.
    '''
    self.subsets.append(self.data)

  def __1by1(self):
    '''
    Create an ensemble with one NN per environment.
    '''
    for env in self.unique_groups:
      only_env = self.data[self.data['1'] == env]
      self.subsets.append(only_env)

  def computeLoss(self):
    '''
    Compute loss of entire ensemble.
    '''
    predictions = np.array(self.predict(self.matrix).detach().numpy())
    mse = np.mean((predictions - self.response)**2)
    self.loss.append(mse)

  def train(self):
    '''
    Train all models.
    '''
    if self.compute_loss:
      # Compute initial loss
      self.computeLoss()
    for i in range(self.num_epochs):
      print(f"Epoch {i+1}")

      for model in self.models:
        model.trainOneEpoch()

      if self.compute_loss:
        self.computeLoss()
        # If loss curve has 'bottomed out', then exit early
        if i > 0 and (self.loss[-1] - self.loss[-2]) > (self.loss[-2] - self.loss[-3]) and self.loss[-1] / self.loss[-2] > 0.95:
          break

  def predict(self, X0):
    '''
    Predict values from matrix X0.
    Computes average of all model predictions.
    '''
    predictions = sum([model.predict(X0) for model in self.models]) / len(self.models)
    return predictions


class NestedCrossValidator:
  '''Class for performing nested cross validation on ensemble neural networks for a CDBN dataset'''
  def __init__(self, k1=5, k2=5, num_epochs=50):
    self.k1 = k1
    self.k2 = k2
    self.num_epochs = num_epochs
    self.subsets = []
    self.current_train_set = None

  def pearsonCorrCoef(self, x1, x2):
    '''x1, x2 column tensors'''
    x1_df = pd.DataFrame(x1.detach().numpy())
    x2_df = pd.DataFrame(x2.detach().numpy())
    return x1_df.corrwith(x2_df)[0]

  def testModel(self, model, test_data):
    '''''
    Test model on test data.
    Returns Pearson correlation coefficient of predictions with true values (testdata).

    model must have predict method
    test_data must be in format: env, genome, yield, snp1, snp2, ...
    '''
    columns_to_drop = ['1','2','3']
    yield_col = '3'
    Xtest = test_data.drop(columns_to_drop, axis = 1)
    ytest = test_data[yield_col]
    predictions = model.predict(Xtest)
    truth = torch.from_numpy(np.array(ytest)) # Ensure that ytest is tensor
    accuracy = self.pearsonCorrCoef(predictions, truth)
    if np.isnan(accuracy):
      print("Nan accuracy found")
    return np.nan_to_num(accuracy)

  def crossValidateParams(self, params, k=None):
    '''
    Score a set of hyperparameters on the current training set
    '''
    if k is None:
      k = self.k2

    shuffled = self.current_train_set.sample(frac=1)
    subsets = np.array_split(shuffled, k)

    scores = []
    for i in range(k):
      test = subsets[i]
      train = pd.concat(subsets[:i] + subsets[i+1:])

      model = Ensemble(train, **params)
      model.train()
      scores.append(self.testModel(model, test))

    return sum(scores)/k

  def createRandomParameterGrid(self, n=100, method="l1o"):
    '''Create a random list of n combinations of hyperparameters'''
    param_grid = []

    for i in range(n):
      lr = np.random.choice(np.logspace(-4, 0))
      batch_size = int(np.random.choice([4,8,16,32,64,128]))
      hidden_dims = []
      depth = np.random.randint(6) + 1
      hidden_dims = []
      for j in range(depth):
        width = np.random.randint(100) + 3
        hidden_dims.append(width)
      
      # Only affects bagging model
      n_weak_learners = np.random.randint(2,30)

      param_grid.append(
          {
              'method': method,
              'lr': lr,
              'batch_size': batch_size,
              'hidden_dims': hidden_dims,
              'num_epochs': self.num_epochs, # maximum number of epochs, will exit early if possible
              'n_weak_learners': n_weak_learners # Only used for bagging
          }
      )
    return param_grid

  def nestedCrossValidation(self, data, k=None, num_params=50, method="l1o", only_do_fold=None, destination="./"):
    '''
    Get unbiased accuracy of Ensemble model using nested cross validation.
    Cross validated model, in each fold hyperparameters are tuned on only train set using further cross validation
    '''
    if k is None:
      k = self.k1

    shuffled = data.sample(frac=1)
    subsets = np.array_split(shuffled, k)
    scores = []

    for i in range(k):
      if only_do_fold is not None and i != only_do_fold:
        continue
      test = subsets[i]
      self.current_train_set = pd.concat(subsets[:i] + subsets[i+1:])

      # Create parameter grid
      param_grid = self.createRandomParameterGrid(n=num_params, method=method)

      print(param_grid)

      # Find model with best hyperparameters
      with Pool() as pool:
        tuning_scores = pool.map(
            self.crossValidateParams, param_grid
        )

      # Save results for inspection later
      results = pd.DataFrame(param_grid)
      results['score'] = tuning_scores
      results.to_csv(f'{destination}/nested_cross_validation_results_{i}.csv')

      # Find best parameters
      best_score = max(tuning_scores)
      index = tuning_scores.index(best_score)
      best_params = param_grid[index]

      # Fit and test model with best hyperparameters
      best_model = Ensemble(self.current_train_set, **best_params)
      best_model.train()
      scores.append(self.testModel(best_model, test))

    # Avg scores
    if only_do_fold is None:
      pd.DataFrame(scores).to_csv(f'{destination}/nested_cross_validation_accuracies.csv')
    else:
      pd.DataFrame(scores).to_csv(f'{destination}/nested_cross_validation_score_{only_do_fold}.csv')
    return sum(scores) / len(scores)

if __name__ == "__main__":
  # Validate command line arguments
  args = sys.argv[1:]
  if len(args) != 4:
    raise ValueError(f"This program requires 4 positional arguments, {len(args)} arguments were provided")
  method, datapath, destination, fold = args
  if method not in ['l1o', 'bagging', 'mlp', '1by1']:
    raise ValueError(f"The method {method} does not exist, please choose from: l1o, bagging, mlp, 1by1")
  if not os.path.exists(datapath):
    raise ValueError(f"The provided dataset: {datapath} does not exist")
  try:
    fold = int(fold)
  except e:
    raise ValueError(f"The provided fold: {fold} is not an integer") 
 
  # Ensure destination folder has been created
  print(f"Method being used: {method}")
  if os.path.isdir(destination):
    print(f"WARNING: Folder /{destination} already exists. Contents may be overridden.")
  else:
    print(f"Creating folder /{destination}")
    os.mkdir(destination)
  
  # Read in data
  data = pd.read_csv(datapath)
  data = data.drop(list(data)[0:2], axis=1) #remove useless columns

  nSNPs = len(data.columns) - 3
  print("there are",len(data["1"].unique()), "locations")
  print("there are", len(data["2"].unique()),"genotypes")
  print("there are",nSNPs, "SNP markers")
  print("there are",len(data),"total observations across all environments")

  # Perform nested cross validation
  cv = NestedCrossValidator(k1=10, k2=3, num_epochs=50)
  accuracy = cv.nestedCrossValidation(data, num_params=300, method=method, destination=destination, only_do_fold=fold)

  print(f"Nested cross validation accuracy: {accuracy}")
