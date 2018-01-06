"""
Train script.
"""
import pandas as pd

from model.diy.model import DecisionTreeID3


def main(path_processed_data,
         model_type):
    """
    Main function training.
    """
    # Load the processed training data
    data_df = load_data(path_processed_data)
    # Init the model
    model = init_model(model_type)
    # Build the model
    model.fit(data_df)
    # Store the model
    # model.store()


def load_data(path_processed_data):
    """Load the processed data."""
    data_df = pd.read_csv(path_processed_data)
    return data_df


def init_model(model_type):
    """Init the model."""
    if model_type == "diy":
        model = DecisionTreeID3()
    return model


if __name__ == '__main__':
    PATH_PROCESSED_DATA = "data/us_election/data.csv"
    MODEL_TYPE = "diy"
    main(PATH_PROCESSED_DATA,
         MODEL_TYPE)
