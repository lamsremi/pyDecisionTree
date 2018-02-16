"""Training script for decision tree.
"""
import importlib

import pandas as pd

import tools


# @tools.debug
def main(train_data=None,
         dataset_name=None,
         model_type=None,
         start_version=None,
         end_version=None,
         **kwargs):
    """Perform a training/fitting/parameters estimation.

    It estimates the parameters of an instance of a model type using
    labeled data.

    Args:
        train_data (iterable type) : the training data. It can be a pandas DataFrame or
        a list of dictionnaries, or a numpy array.
        dataset_name (str): name of the dataset which is the name of the folder in data directory.
        model_type (str): type of model to train.
        start_version (str): starting version.
        end_version (str): name to identify the version to store the fitted parameters.

    Returns:
        output (str): Different indication messages.
    """
    # If no data is provided
    if train_data is None:
        # Load labaled data from a dataset
        train_data = load_train_data(dataset_name)

    # Initialize an instance of the class corresponding to the given type of model
    model = initialize_model(model_type)

    # Load the initial parameters of the instance using the given starting version
    model.load_parameters(model_version=start_version)

    # Fit the model
    model.fit(train_data=train_data)

    # Persist the parameters of the model
    model.persist_parameters(model_version=end_version)


# @tools.debug
def load_train_data(dataset_name):
    """Load labeled data from data directory.

    Args:
        dataset_name (str): dataset to load.

    Return:
        train_data (iterable type) : the training data. It can be a pandas DataFrame or
        a list of dictionnaries, or a numpy array.
    Note:
        a dataset is defined by his "dataset_name" name which is the name
        of the folder in data directory.
    """
    # Load data using pandas library into a DataFrame
    data_df = pd.read_pickle("data/{}/data.pkl".format(dataset_name))
    # Format the data into the required format of the fit model
    train_data = format_data(data_df)
    # Return the training data
    return train_data


# @tools.debug
def format_data(data_df):
    """Convert the data into the proper format_dataat.
    Args:
        data_df (DataFrame): table of the training data.
    Return:
        train_data (iterable type) : the training data. It can be a pandas DataFrame or
        a list of dictionnaries, or a numpy array.
    """
    ## Example ##############################
    # train_data = []
    # for row in data_df.itertuples():
    #     train_data.append([value for value in row[1:]])
    #########################################
    # Return the table
    return train_data


def initialize_model(model_type):
    """ Initialize an instance of a model.

    Args:
        model_type (str): type of the model to init.

    Return:
        model (object): initialized instance of the given type of model.
    """
    # Import the module from the library
    model_class = importlib.import_module("library.{}.model".format(model_type))
    # Initialize the instance
    model = model_class.Model()
    # Return the initialized instance
    return model


# To use only for development
# if __name__ == '__main__':
#     TRAIN_DATA = blabla
#     main(train_data=TRAIN_DATA,
#          dataset_name=None,
#          model_type="diy",
#          start_version=None,
#          end_version="X")
