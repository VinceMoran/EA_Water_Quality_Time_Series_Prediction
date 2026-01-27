
# Import the relevant libraries
from pathlib import Path
import os
import zipfile
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import inspect
import torch
from torch import nn
from torch.nn import Module
import collections
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
import hashlib
import re
import torch.nn.functional as F
from IPython import display
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# Define helper function for reshaping tensors (source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)

# Define a helper function for changing tensor data types (source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

# Define a helper function for returning the indices of all elements in a tensor (source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)

# Compute the mean average of a tensor by reducing specified dimensions (or all if no dimensions specified)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)

# Swap two axes of a tensor as specified
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)

# Detach a PyTorch Tensor from GPU (if required) and convert to NumPy ndarray
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)

# Move a PyTorch Tensor to a different device (and change data type if specified)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)


def load_raw_data(url: str, data_dir: str = "data/raw_data") -> Path:
    """
    Downloads a ZIP archive from a URL, extracts its contents to a target directory,
    and deletes the ZIP file.

    Args:
        url (str): URL of the ZIP archive containing the raw data.
        data_dir (str): Directory to extract the data into (default: "data/raw_data").

    Returns:
        Path: Path to the directory containing the extracted files.
    """
    data_path = Path(data_dir)

    # Create directory if it does not exist
    if data_path.is_dir():
        print(f"[INFO] {data_path} directory already exists.")
    else:
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] {data_path} directory has been created.")

    zip_path = data_path / "raw_water_quality_parameter_data.zip"

    # Download the ZIP file
    print("[INFO] Downloading data...")
    response = requests.get(url)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract ZIP file
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
            print(f"[INFO] Data extracted to {data_path}.")
    except zipfile.BadZipFile:
        raise RuntimeError(f"[ERROR] The file at {zip_path} is not a valid ZIP file.")

    # Remove ZIP file
    os.remove(zip_path)
    print("[INFO] ZIP file removed.")

    return data_path


def cpu():
    """
    Get the CPU device.

    Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    return torch.device('cpu')

def gpu(i=0):
    """
    Get a GPU device.

    Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    return torch.device(f'cuda:{i}')


def count_gpus():
    """
    Get the number of available GPUs.

    Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    return torch.cuda.device_count()

# Define a function for computing performance metrics
def compute_performance_metrics(y_obs: np.ndarray,
                              y_pred: np.ndarray):
  """Compute performance metrics for model predictions.

  Args:
    y_obs (np.ndarray): Observed values.
    y_pred (np.ndarray): Predicted values.

  Returns:
    metrics (dict): A dictionary of performance metrics.
  """

  # Ensure correct sequence lengths
  if len(y_obs) != len(y_pred):
    raise ValueError("Input arrays must have the same length.")

  # Compute root mean squared error
  rmse = root_mean_squared_error(y_true=y_obs,
                                 y_pred=y_pred)

  # Compute mean absolute error
  mae = mean_absolute_error(y_true=y_obs,
                               y_pred=y_pred)

  # Compute coefficienct of determination
  r_squared = r2_score(y_true=y_obs,
                       y_pred=y_pred)

  # Compute the Nash-Sutcliffe efficiency
  nse = 1 - (np.sum((y_obs - y_pred)**2) / np.sum((y_obs - np.mean(y_obs))**2))

  # Store metrics in a dictionary
  metrics = {"RMSE": rmse,
             "MAE": mae,
             "r^2": r_squared,
             "NSE": nse}

  return metrics

def use_svg_display():
    """
    Use the svg format to display a plot in Jupyter.

    Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    # Instruct Jupyter to render all plots in SVG format instead of PNG (default).
    backend_inline.set_matplotlib_formats('svg')

# Define a function for computing the unqiue value counts of variables
def unique_value_counts(dataframe: pd.DataFrame,
                        dataframe_name: str):
  """
  Takes in a pandas DataFrame and prints the unique value counts for each
  variable.

  Args:
    dataframe (pd.DataFrame): A pandas DataFrame to be investigated.
    dataframe_name (str): The name of the DataFrame to be printed for clarity.

  Returns:
    None.
  """
  # Print the output header
  print(f"DISTINCT VALUE COUNTS FOR {dataframe_name}:")
  print("{:16} {:^25}".format("Variable", "Number of Distinct Value Counts")) # print a header row for the table

  # Loop through the DataFrame columns and print a row for each variable
  for variable in dataframe.columns:
    print("{:16} {:^25}".format(variable, dataframe[variable].nunique()))  # Use nunique() to get unique value count
    print()

# Define a function for computing datatypes and checking missing values
def datatypes_and_missing_values(dataframe: pd.DataFrame,
                                 dataframe_name: str):
    """
    Takes in a pandas DataFrame and prints the datatype and number of missing
    values for each variable.

    Args:
      dataframe (pd.DataFrame): A pandas DataFrame to be investigated.
      dataframe_name (str): The name of the DataFrame to be printed for clarity.

    Returns:
      None.
    """
    # Print the output header
    print(f"DATATYPES AND MISSING VALUES FOR {dataframe_name}")
    print("{v: <18} {dt: <14} {n: ^24}".format(v="Variable", dt="Data Type", n="Number of Missing Values")) # Print a header row for the table.

    # Loop through the DataFrame columns and print a row for each variable
    for variable in dataframe.columns:
        print("{v: <18}, {dt: <14}, {n: ^24}".format(v=variable, dt=str(dataframe[variable].dtype), n=dataframe[variable].isnull().sum()))

# Create a function for temporally splitting data into training, validation, and testing sets
def train_val_test_split(time_series: pd.DataFrame,
                        train_size: float,
                        val_size: float,
                        test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Splits a time series dataset into training, cross-validation, and testing sets
  based on specified proportions without shuffling.

  Args:
    time_series (pd.DataFrame): A DataFrame containing time series data.
    train_size (float): Proportion of data to use for training
    val_size (float): Proportion of data to use for validation and grid search.
    test_size (float): Proportion of data to withhold for testing.

  Returns:
    tuple(train_set, val_set, test_set): Tuple that contains DataFrames containing training,cross validation, and testing data.
  """

  # Enforce the condition that all proportions add to 100% (allowing for floating point tolerance during summation)
  if not np.isclose(train_size + val_size + test_size, 1.0):
    raise ValueError("Training, cross-validation, and testing proportions must sum to equal 1.")

  # Compute number of samples in each set
  n_samples = len(time_series)
  n_train = int(round(train_size * n_samples))
  n_val = int(round(val_size * n_samples))

  # Split the data into training, cross-validation, and testing sets
  train_set = time_series[:n_train]
  val_set = time_series[n_train:n_train+n_val]
  test_set = time_series[n_train+n_val:]

  return train_set, val_set, test_set


# Create a base class for hyperparameters to streamline optimisation
class HyperParameters:
    """The base class of hyperparameters.
    Sourced from `d2l.torch`:
    https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py

    Provides utlity functions to automatically save function arguments as
    hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """
        Placeholder function for saving hyperparameters: Not defined here.

        This method should be implemented within subclasses that
        inherit from the HyperParameters base class."""
        # Functionality to be overwritten to indicate the above description
        raise NotImplemented

    def save_hyperparameters(self,
                             ignore: list = []):
        """Saves caller function arguments as class attributes. Arguments
        listed in `ignore` are excluded. Attributes are also stored in the
        `self.hparams` dictionary for reference.

        Args:
          self:
          ignore (list, optional): list of hyperparameter names to ignore. Defaults to [].

        Returns:
          None.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(HyperParameters):
    """
    Animated progress board for plotting data points when training and
    evaluating models.

    Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    # Call the constructor method and save arguments as class attributes
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self,
             x: float,
             y: float,
             label: str,
             every_n: int=1):
        """
        Plots and animates data points in Jupyter during training/evaluation.

        Args:
          - x (float): The x-coordinate of the data point.
          - y (float): The y-coordinate of the data point.
          - label (str): The label for the curve that the data point belongs to.
          - every_n (int): Plot every `every_n` data points for smoothing curves.

        Side effects:
          - Plots every nth data point on the progress board.
        """
        # Create a named Tuple to store data points
        Point = collections.namedtuple('Point', ['x', 'y'])
        # Initialise storage for raw points and data on the first function call
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        # Create empty lists to store points and data for each label
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        # Add the new points and data to raw points and data dictionaries for each label
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        # Create condition that every_n points have been collected before averaging and plotting
        if len(points) != every_n:
            return
        # Smooth data by averaging the last every_n points
        mean = lambda x: sum(x) / len(x)
        # Add the averaged data to the data points that will be plotted
        line.append(Point(mean([p.x for p in points]),
                          mean([p.y for p in points])))
        # Clear the points to prepare for the next batch
        points.clear()
        # If argument `display=False` then skip plotting
        if not self.display:
            return
        # Make plots sharper and scalable in Jupyter notebooks
        use_svg_display()
        # Create a figure if none is specified in the constructor
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        # Plot all lines using their saved linestyles and colours and save for legend
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                      linestyle=ls, color=color)[0])
            labels.append(k)
        # Uses axes specified in the constructor or defaults to `plt.gca()` if None
        axes = self.axes if self.axes else plt.gca()
        # Sets axis limits
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        # Sets axis labels
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        # Sets axis scales
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        # Create legend
        axes.legend(plt_lines, labels)
        # Display the figure in Jupyter
        display.display(self.fig)
        # Enable animation effect by replacing old frames
        display.clear_output(wait=True)


class DataModule(HyperParameters):
    """
    The base class of data.

    Inherits from HyperParameters.

    Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    def __init__(self,
                 num_workers=4):
        self.save_hyperparameters()

    # Assert that subclasses inheriting from DataModule must implement a `get_dataloader` method
    def get_dataloader(self, train):
        # Raises `NotImplementedError` indicating that this base class cannot be used directly
        raise NotImplementedError

    # Define a method for returning a training DataLoader
    def train_dataloader(self):
        return self.get_dataloader(train=True) # Defaults to `train=True` for shuffling

    # Define a method for returning a validation DataLoader
    def val_dataloader(self):
        return self.get_dataloader(train=False) # Defaults to `train=False` for NO shuffling

    # Define a method for creating a PyTorch DataLoader from raw Tensors
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """
        Creates a PyTorch DataLoader from raw Tensors.

        Args:
          - tensors (Tuple[Tensor, ...]): Tuple of tensors with matching first
          dimensions in the format (X, y) or similar.
          - train (bool): Whether the DataLoader is used for training or
          validation (controls shuffling).
          - indices (slice or list, optional): Subset of the dataset that
          should be sampled (defaults to  `slice(0, None)` to sample the whole
          dataset.)

        Returns:
          - DataLoader: A PyTorch DataLoader object.
        """
        # Apply the indices slice to select a subsampled Tuple of tensors
        tensors = tuple(a[indices] for a in tensors)
        # Create a PyTorch TensorDataset object where each element combines corresponding elements from the input tensors
        dataset = torch.utils.data.TensorDataset(*tensors)
        # Wrap the TensorDataset object in an iterable PyTorch DataLoader and return
        return torch.utils.data.DataLoader(
            dataset, # the TensorDataset object to be wrapped in an iterable DataLoader
            self.batch_size, # Determines how many samples should be included in each minibatch
            shuffle=train # Applies Train=True/False to determine whether data should be shuffled
            )


class TimeSeries(DataModule):
    """
    Class for the Time Series Water Quality Dataset.

    Inherits from DataModule.

    Modified From Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    # Initialise the dataset
    def __init__(self,
                 batch_size, # number of sequences per minibatch
                 num_steps, # sequence length for input/output pairs
                 X_train, # Array of features for training
                 y_train, # Array of targets for training
                 X_val, # Array of features for validation
                 y_val, # Array of targets for validation
                 X_test, # Array of features for testing
                 y_test # Array of targets for testing
                 ):
        super(TimeSeries, self).__init__()
        self.save_hyperparameters()

        # Convert the input arrays into PyTorch Tensors with sliding windows for LSTM inputs
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unfold(0, num_steps, 1)[:-1].permute(0, 2, 1)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)[num_steps:]
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unfold(0, num_steps, 1)[:-1].permute(0, 2, 1)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)[num_steps:]
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unfold(0, num_steps, 1)[:-1].permute(0, 2, 1)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)[num_steps:]

    # Return train DataLoader
    def train_dataloader(self):
        return self.get_tensorloader([self.X_train_tensor, self.y_train_tensor], train=True)

    # Return validation DataLoader
    def val_dataloader(self):
        return self.get_tensorloader([self.X_val_tensor, self.y_val_tensor], train=False)

    # Return test DataLoader
    def test_dataloader(self):
        return self.get_tensorloader([self.X_test_tensor, self.y_test_tensor], train=False)


class Module(nn.Module, HyperParameters):
    """
    The base class for neural network models that includes hyperparameter
    management and progress plotting.

    Inherits from `torch.nn.Module` and `HyperParameters`.

    Modified from Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    # Assert that classes that inherit from Module must implement a loss function
    def loss(self, y_hat, y):
        raise NotImplementedError

    # Perform the forward computation and assert that the `self.net` attribute must exist
    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    # Define the function for plotting progress in training/evaluation
    def plot(self, key, value, train):
        """Plot a point in animation.

        Args:
          - key (str): The name of the key to plot.
          - value (float): The value to plot.
          - train (bool): True indicates that training data should be plotted,
          whilst False indicates that validation data should be plotted.

        Side effects:
          - Updates and plots a progress metric on the board.
        """
        # Assert that a trainer instance must exist
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        # Set the x axis label as the number of training epochs
        self.board.xlabel = 'epoch'
        # Determine x co-ordinates and aggregation frequency if training data
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        # Determine x co-ordinates and aggregation frequency if validation data
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        # Draw the points and set label as specified to training/validation
        self.board.draw(x, numpy(to(value, cpu())),
                ('train_' if train else 'val_') + key,
                every_n=int(n))

    # Computes loss and plots metrics for a single training step
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    # Computes loss and plots metrics for a single validation step
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    # Returns optimiser for training the model
    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return torch.optim.Adam(self.parameters(), # Defaults to Adam optimiser using saved model parameters
                                lr=self.lr, # Uses learning rate = `self.lr`
                                weight_decay=self.weight_decay # Uses weight decay regularisation rate = `self.weight_decay
                                )

    # Perform a forward pass to initialise the network, and applies weight initialisation if specified (optional)
    def apply_init(self, inputs, init=None):
        """Defined in :numref:`sec_lazy_init`"""
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


class RNN(Module):
    """
    Base class for RNN models implemented with high level APIs.

    Inherits from Module.

    Modified from Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    # Initialise the RNN class with hyperparameters `num_inputs`, `num_hiddens`, `num_layers`, and `dropout`
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout):
        super().__init__()
        # Save hyperparameters `num_inputs` and `num_hiddens` as class attributes
        self.save_hyperparameters()
        # Define the RNN model using `torch.nn.RNN`
        self.rnn = nn.RNN(input_size=num_inputs,
                          hidden_size=num_hiddens,
                          num_layers=num_layers,
                          dropout=dropout, # uses L1 regularisation at specified rate
                          batch_first=True)

    # Define the forward computation
    def forward(self, inputs, H=None):
        #Perform the forward pass through the RNN
        return self.rnn(inputs, H)


# Define the long short-term memory (LSTM) class
class LSTM(RNN):
  """
  Class for implementing Long Short-Term Memory (LSTM) models.

  Inherits from RNN base class.
  """
  # Initialise the LSTM class with hyperparameters `num_inputs` and `num_hiddens`
  def __init__(self, num_inputs, num_hiddens, num_layers, dropout):
    # Call the constructor of the Module base class
    Module.__init__(self)
    # Save hyperparameters `num_inputs` and `num_hiddens` as class attributes
    self.save_hyperparameters()
    # Define the LSTM model using `torch.nn.LSTM`
    self.rnn = nn.LSTM(input_size=num_inputs,
                       hidden_size=num_hiddens,
                       num_layers=num_layers,
                       dropout=dropout,
                       batch_first=True)

  # Define the forward computation using `self.RNN`
  def forward(self, inputs, H_C=None):
    # Perform the forward pass through the LSTM
    return self.rnn(inputs, H_C)


class Trainer(HyperParameters):
    """The base class for training models with data.

    Inherits from HyperParameters.

    Modified from Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py"""

    # Call the class constructor, save hyperparameters, and use GPU support if available
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        """
        Init the Trauiner class using device agnostic code to use GPU support if available.

        Args:
          - max_epochs (int): The maximum number of epochs to train the model for.
          - num_gpus (int): The number of GPUs to use if possible (default=0).
          - gradient_clip_val (float): The maximum gradient value to clip to prevent explosions (default=0).

        Side effects:
          - Calls the `save_hyperparameters` method to save the hyperparameters of the model.
          - Sets the number of GPUs for use as the minimum of the number specified and number available.
          - Sets the `gpus` attribute to a list of available GPUs where the list is empty if none are available.
        """
        self.save_hyperparameters()
        # Create a list containing the available and requested GPUs (empty if none both available & requested)
        self.gpus = [gpu(i) for i in range(min(num_gpus, count_gpus()))] # Use as many GPUs as requested provided they are also available to use


    # Prepare data for modelling by taking a data object
    def prepare_data(self, data):
        # Save a training DataLoader object as a class attribute
        self.train_dataloader = data.train_dataloader()
        # Save a validation DataLoader object as a class attribute
        self.val_dataloader = data.val_dataloader()
        # Save a test DataLoader object as a class attribute
        self.test_dataloader = data.test_dataloader()
        # Save the length of the training DataLoader as a class attribute
        self.num_train_batches = len(self.train_dataloader)
        # Save the length of the validation DataLoader as a class attribute
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        # Save the length of the test DataLoader as a class attribute
        self.num_test_batches = (len(self.test_dataloader)
                                if self.test_dataloader is not None else 0)


    def prepare_model(self, model):
        """
        Prepare a model for device agnostic training.

        Args:
          - model (Module): The model to be trained.

        Side effects:
          - Links a model instance to a trainer instance so that they can share information.
          - Moves the model to a GPU if available.
        """
        # Save an instance of the Trainer class as a model attribute to share information
        model.trainer = self
        # Set the xlim attribute of the model progress board to the max number of epochs
        model.board.xlim = [0, self.max_epochs]
        # Move the model to a GPU if available
        if self.gpus:
            model.to(self.gpus[0])
        # Save an instance of the model as an attribute of the Trainer class to share information
        self.model = model


    def fit(self, model, data):
        """
        The primary method for fitting a model to data.

        Args:
          - model (Module): The model to be trained.
          - data (DataModule): The data to be used for training and validation.

        Side effects:
          - Calls the `prepare_data` method to prepare the data for training.
          - Calls the `prepare_model` method to prepare the model for training.
          - Calls the `configure_optimizers()` method to configure the optimiser.
          - Initialises the `epoch` attribute as zero before training the model.
          - Initialises the `train_batch_idx` attribute as zero before training the model.
          - Initialises the `val_batch_idx` attribute as zero before training the model.
          - Loops through calls to the `fit_epoch` method to train the model for the specified number of epochs.
        """
        # Prepare the data for training
        self.prepare_data(data)
        # Prepare the model for training
        self.prepare_model(model)
        # Configure optimizers
        self.optim = model.configure_optimizers()
        # Initialise epochs and batch numbers to zero before training/validation
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        # Loop through the number of training/validation epochs
        for self.epoch in range(self.max_epochs):
            # Fit the model for a single epoch
            self.fit_epoch()

    def prepare_batch(self, batch):
        """
        Prepare batches for device agnostic training.

        Args:
          - batch (Iterable[Tensor]): The batch of data to be prepared.

        Returns:
          - batch (Tensor): The prepared batch.

        Side effects:
          - Move the batch to a GPU if available.
        """
        if self.gpus:
            batch = [to(a, self.gpus[0]) for a in batch]
        return batch


    def fit_epoch(self):
        """
        Fits the model to data for a single epoch.

        Side effects:
          - Calls the `train_batch` method to train the model for a single epoch.
        """
        self.model.train()
        for batch in self.train_dataloader:
            loss = self.model.training_step(self.prepare_batch(batch))
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:  # To be discussed later
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()
            self.train_batch_idx += 1
        if self.val_dataloader is None:
            return
        self.model.eval()
        for batch in self.val_dataloader:
            with torch.no_grad():
                self.model.validation_step(self.prepare_batch(batch))
            self.val_batch_idx += 1

    def clip_gradients(self, grad_clip_val, model):
        """
        Clips gradients to prevent instability from exploding gradients if specified.

        Args:
          - grad_clip_val (float): The maximum gradient value to clip to.
          - model (Module): The model to be trained.

        Side effects:
          - Clips the gradients of the model parameters to remain below a maximum value if specified.
        """
        # Obtain all model parameters where gradient tracking is enabled
        params = [p for p in model.parameters() if p.requires_grad]
        # Compute the norm of all tracked gradients
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params if p.grad is not None))
        # Clip gradients where the norm exceeds the clipping value
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm


class Regressor(Module):
    """
    The base class of regression models.

    Inherits from Module.

    Modified from Source: https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """

    def validation_step(self, batch):
        """
        Defines a method for evaluating the model on evaluation data.

        Args:
          - batch (Iterable[Tensor]): The batch of data to be evaluated.

        Side effects:
          - Calculates and plots the loss and accuracy of the model on the evaluation data.
        """
        # Slice all data except the last element (class label)
        Y_hat = self(*batch[:-1]) # Self calls the model forward pass to obtain prediction logits/probabilities/labels depending on model
        # Plot loss on evaluation data
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        # Plot accuracy on evaluation data
        self.plot('MAE', self.mean_absolute_error(Y_hat, batch[-1]), train=False)

    def mean_absolute_error(self, Y_hat, Y):
        """
        Defines a method for computing mean absolute error (MAE) of model
        predictions on evaluation data.

        Args:
          - Y_hat (Tensor): The model predictions.
          - Y (Tensor): The true labels.

        Returns:
          - The MAE of the model predictions as a scalar floating point value.
        """
        # Compute the MAE
        MAE = torch.mean(torch.abs(Y_hat - Y))
        # Returns the scalar mean if `averaged=True` or the per-element loss vector if `averaged=False`
        return MAE # May need to swap to `return MAE.item() if unstable for plotting code.

    def loss(self, Y_hat, Y, averaged=True):
        """
        Defines a method for computing the loss of model predictions on
        evaluation data (defaults to mean squared error for regression tasks to
        increase likelihood of stable optimisation).

        Args:
          - Y_hat (Tensor): The model predictions.
          - Y (Tensor): The ground truth values.
          - averaged (bool): Whether to average the loss across batches or return vectors containing per-sample loss metrics (default=True).

        Returns:
          - The loss of the model predictions as a floating point scalar (`averaged=True`) or vector (`averaged=False`).
        """
        # Return the output of the MAE loss function as a scalar or vector as specified
        return F.mse_loss(Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """
        Defines a method for printing/inspecting layer output shapes.

        Args:
          - X_shape (tuple): The shape of the input data.

        Side effects:
          - Prints the output shape of each layer in the model.
        """
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class RNNWQ(Regressor):
    """
    The base class for RNN-based water quality models.

    Inherits from Regressor.

    Modified from Source Class (RNNLM):
    https://raw.githubusercontent.com/d2l-ai/d2l-en/refs/heads/master/d2l/torch.py
    """
    # Call the class constructor and save hyerparameters
    def __init__(self,
                 rnn,
                 out_features,
                 lr=0.01, # Learning rate defaults to 0.01
                 weight_decay=0.0001 # Weight decay regularisation rate defaults to 0.0001
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
      """
      Define a method for initialising model parameters.

      Args:
        - None.

      Returns:
        - None:

      Side effects:
        - Initialises a trainable linear layer with output shape [vocab_size]
      """
      # Define a linear layer with `out_features=self.out_features` and `in_features` inferred automatically
      self.linear = nn.LazyLinear(self.out_features)


    def training_step(self, batch):
        """
        Defines a method for computing and plotting loss on training data.

        Args:
          - batch (Iterable[Tensor]): Batch of data containing multiple tensors of form [X1, X2, X3, y].

        Returns:
          - l (Tensor): The loss of the model predictions on the data.

        Side effects:
          - Plots the loss of the model in predicting samples from the training data.
        """
        # Compute the loss of the model by calling the forward pass on the data `self(*batch[:-1]) and comparing with labels `batch[-1]
        l = self.loss(self(*batch[:-1]), batch[-1])
        # Plot loss for each for training step
        self.plot('loss', l, train=True)
        # Return the computed loss for further use
        return l

    def validation_step(self, batch):
        """
        Defines a method for computing and plotting loss on validation data.

        Args:
          - batch (Iterable[Tensor]): Batch of data containing multiple tensors of form [X1, X2, X3, y].

        Returns:
          - None.

        Side effects:
          - Computes the loss of the model on the evaluation data.
          - Plots the loss of the model in predicting samples from the evaluation data.
        """
        # Compute the loss of the model using the same method from `training_step`
        l = self.loss(self(*batch[:-1]), batch[-1])
        # Plot the loss for each evaluation step
        self.plot('loss', l, train=False)

    def output_layer(self, hiddens):
        """
        Defines a method for computing the outputs from the final layer of the
        model.

        Args:
          - hiddens (Tensor): The outputs from the RNN hidden layers.

        Returns:
          - (Tensor): The outputs from the final layer of the model.

        Side effects:
          - None.
        """
        # Pass the outputs of the RNN through a linear output layer
        return self.linear(hiddens) # Shape: (batch_size, seq_length, out_features)

    def forward(self, X, state=None):
        """
        Defines a method for computing the forward pass through the model.

        Args:
          - X (Tensor): The input tensor.
          - state (Tensor): The initial state of the RNN (default=None).

        Returns:
          - The outputs from the final layer of the model
        """
        # Pass the input tensor through the RNN taking `initial state = state`
        rnn_outputs, _ = self.rnn(X, state) # Outputs `rnn_outputs` which is the sequence of RNN hidden states over time (second output for final hidden state is not needed so placeholder `_` is used)
        # Obtain the final values for hidden layers to pass for next step prediction at the final timestep
        rnn_final_output = rnn_outputs[:, -1, :] # Takes only the final timestep for full batch and output size
        # Return the outputs after passing the last RNN hidden state through the final output layer
        return self.output_layer(rnn_final_output)

    def predict(self, X, seq_length=96, batch_size=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Defines a method for producing model predictions at the specified
        timestep (recursive forecasting or returning intermediate predictions
        for mutlti-step targets is not enabled by this model as the number of
        unknown input features exceeds the number of known targets for
        recursive forecasts). Targets beyond t+1 can still be used through
        manual engineering of target tensors and retraining the model (e.g.
        retraining for t+96 targets).

        Args:
          - X (np.ndarray): The input features [X1, X2, X3,..., Xn] with shape [time_steps, num_features].
          - seq_length (int): The length of the sequence to be fed to the RNN model (default=96).
          - batch_size (int): Batch size used for inference (default=1024).
          - device (str): The device on which to run inference (defaults to GPU if available).

        Returns:
          - y_preds (np.ndarray): Predicted values of shape [num_sequences,].
        """
        self.to(device)
        self.eval()

        # Convert input to tensor of the correct dimensions with sequences of the correct length
        X_tensor = torch.tensor(X, dtype=torch.float32).unfold(0, seq_length, 1)[:-1].permute(0, 2, 1) # [num_sequences, seq_length, features]

        # Create an empty list to hold predictions
        preds = []

        # Turn off gradient tracking for inference
        with torch.no_grad():
            # Loop through batches for inference
            for i in range(0, len(X_tensor), batch_size):
                # Obtain the current batch and send to same device as model
                batch = X_tensor[i:i+batch_size].to(device)
                # Perform the forward pass through the model to obtain predictions for current batch
                y_batch = self.forward(batch) # dims: [batch_size, 1]
                # Append the predictions from the current batch to the list
                preds.append(y_batch.cpu())
        # Return the predictions as a NumPy ndarray
        return torch.cat(preds, dim=0).squeeze(-1).numpy()
