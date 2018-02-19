"""Preprocessing of the data

The raw data is a dataset of the us election

source: https://github.com/saimadhu-polamuri/DataAspirant_codes/
tree/master/Logistic_Regression/Logistic_Binary_Classification

Descriptions:
- 944 rows
- X columns:
    * popul
    * TVnews
    * selfLR
    * ClinLR
    * DoleLR
    * PID
    * age
    * educ
    * income
    * vote
"""
import pickle

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.width', 800)


def main():
    """
    Preprocess the data.
    """
    # Load the raw data
    raw_data_df = load_raw_data("data/us_election/raw_data/data.csv")
    # Study data
    study_data(raw_data_df, data_type="numerical")
    # Transform the data
    data_df = process(raw_data_df)
    # Study transformed data
    study_data(data_df, data_type="categorical")
    # Format the data
    train_data = format_data(data_df)
    # Store the data
    store(train_data, "data/us_election/data.pkl")


def load_raw_data(path_raw_data):
    """Load the raw data."""
    raw_data_df = pd.read_csv(
        path_raw_data,
    )
    return raw_data_df


def study_data(data_df, data_type):
    """
    Examine the data.
    """
    # Display shape
    print("- shape :\n{}\n".format(data_df.shape))
    # Display data dataframe (raws and columns)
    print("- dataframe :\n{}\n".format(data_df.head(100)))
    # Display types
    print("- types :\n{}\n".format(data_df.dtypes))
    # Missing values
    print("- missing values :\n{}\n".format(data_df.isnull().sum()))
    if data_type == "numerical":
        # Display value distribution
        # display_histogram(data_df)
        # Extreme values
        print(" - minimum values :\n{}\n".format(data_df.min()))
        print(" - maximum values :\n{}\n".format(data_df.max()))
    elif data_type == "categorical":
        # Unique value
        print("- unique values :")
        for attribute in list(data_df.columns):
            print("  * {} : {}".format(attribute, data_df[attribute].unique()))


def display_histogram(data_df):
    """Display historgram."""
    plt.figure()
    for attribute in list(data_df.columns):
        data_df[["{}".format(attribute)]].plot.hist(bins=20)
        plt.savefig("data/us_election/raw_data/histogram_{}.png".format(attribute))


def process(raw_data_df):
    """
    Process the data so it can be used by the mdoel
    """
    # Categorize the numerical variables
    data_df = categorize_frame(raw_data_df)
    # Convert output to string
    data_df["vote"] = data_df["vote"].apply(lambda x: str(x))
    return data_df


def categorize_frame(data_df):
    """
    Process
    """
    resolution_dict = {
        "popul": 2000,
        "TVnews": 3,
        "selfLR": 3,
        "ClinLR": 3,
        "DoleLR": 3,
        "PID": 2,
        "age": 30,
        "educ": 3,
        "income": 10
    }
    for attribute, resolution in resolution_dict.items():
        data_df[attribute] = categorize_serie(data_df[attribute], resolution)
    return data_df


def categorize_serie(data_serie, resolution):
    """
    Categorize serie given a resolution.
    """
    data_serie = data_serie.apply(
        categorize,
        args=(resolution,)
    )
    return data_serie


def categorize(x_value, resolution):
    """Categorize function."""
    output = "{}_{}".format(x_value//resolution*resolution,
                            (x_value//resolution+1)*resolution)
    return output


def format_data(data_df):
    """Format the data.
    """
    train_data = []
    for row in data_df.itertuples():
        train_data.append([value for value in row[1:]])
    return train_data


def store(train_data, path_preprocessed_data):
    """Store the processed data."""
    with open(path_preprocessed_data, "wb") as handle:
        pickle.dump(train_data, handle)
