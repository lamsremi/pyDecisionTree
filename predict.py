"""Predict script for the given task.

This module is used for performing a prediction.
"""
import pickle
import importlib

import tools

@tools.debug
def main(inputs_data=None,
         model_type=None,
         model_version=None,
         model=None):
    """Perform a prediction.

    Args:
        inputs_data (iterable type) : list of inputs to predict.
        model_type (str): type of model to use for prediction.
        model_version (str): version of the model to use.
        model (Object): model instance from class Model previously loaded.
        To be use for optimization purpose.

    Returns:
        outputs_data (iterable type): list of predicted outputs in the same order than the inputs.
    """
    # If no loaded instance is given
    if model is None:
        # Initialize an instance of the class corresponding to the given type of model
        model = initialize_model(model_type)

    # Load the parameters of the model given the version.
    model.load_parameters(model_version=model_version)

    # Perform the prediction
    outputs_data = model.predict(inputs_data)

    # Return table
    return outputs_data


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

    with open("data/us_election/data.pkl", "rb") as handle:
        inputs_data = [row[:-1] for row in pickle.load(handle)][0:5]
    for s in inputs_data:
        print(s)
    main(inputs_data=inputs_data,
         model_type="python_CART",
         model_version="us_election",
         model=None)
