"""Training script for decision tree.
"""
import pickle
import importlib

import tools


# @tools.debug
def main(train_data=None,
         header = None,
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
    if train_data is None or header is None:
        # Load labaled data from a dataset
        train_data, header = load_train_data(dataset_name)

    # Initialize an instance of the class corresponding to the given type of model
    model = initialize_model(model_type)

    # Load the initial parameters of the instance using the given starting version
    model.load_parameters(model_version=start_version)

    # Fit the model
    model.fit(train_data=train_data[0:60], header=header)

    # Persist the parameters of the model
    model.persist_parameters(model_version=end_version)

    # Display tree
    model.display_tree()

# @tools.debug
def load_train_data(dataset_name):
    """Load labeled data from data directory.

    Args:
        dataset_name (str): dataset to load.

    Return:
        train_data (iterable type) : the training data. It can be a pandas DataFrame or
        a list of dictionnaries, or a numpy array.
        header
    Note:
        a dataset is defined by his "dataset_name" name which is the name
        of the folder in data directory.
    """
    # Load data using pickle
    with open("data/{}/data.pkl".format(dataset_name), "rb") as handle:
        train_data = pickle.load(handle)
    # Load header
    with open("data/{}/header.pkl".format(dataset_name), "rb") as handle:
        header = pickle.load(handle)
    # Return the training data
    return train_data, header


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
if __name__ == '__main__':
    main(train_data=None,
         header=None,
         dataset_name="us_election",
         model_type="python_CART",
         start_version=None,
         end_version="us_election")
