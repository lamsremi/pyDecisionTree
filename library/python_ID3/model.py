"""Implementation of a Decision Tree Classifier in pure python.

Ref:
* https://en.wikipedia.org/wiki/ID3_algorithm
* https://www.youtube.com/watch?v=LDRbO9a6XPU
"""
import numpy as np
import pandas as pd

import tools

class Model():
    """Decision tree model
    """

    def __init__(self):
        """
        Initialize an instance.
        """
        self.model = {}


    # def init_model(self):
    #     """Inits the model weights."""
    #     model = {
    #         "attribute": "salary",
    #         "categories": [
    #             {
    #                 "value": "1000_5000",
    #                 "tree": {
    #                     "attribute": "time",
    #                     "categories": [
    #                         {
    #                             "value": "0_10",
    #                             "tree": {
    #                                 "label": "1"
    #                             }
    #                         },
    #                         {
    #                             "value": "10_50",
    #                             "tree": {
    #                                 "attribute": "free",
    #                                 "categories": [
    #                                     {
    #                                         "value": "true",
    #                                         "tree": {
    #                                             "label": "0"
    #                                         }
    #                                     },
    #                                     {
    #                                         "value": "false",
    #                                         "tree": {
    #                                             "label": "1"
    #                                         }
    #                                     }
    #                                 ]
    #                             }
    #                         }
    #                     ]
    #                 }
    #             },
    #             {
    #                 "value": "0_1000",
    #                 "tree": {
    #                     "label": "0"
    #                 }
    #             }
    #         ]
    #     }
    #     return model

    # @tools.debug
    def predict(self, input_data=None):
        """
        Predict.
        """
        if input_data is None:
            input_data = {
                "salary": "1000_5000",
                "time": "10_50",
                "free": "false"
            }
        label = None
        model_cursor = self.model

        i_value = 1
        while label is None:
            print("level - {}".format(i_value))
            if "label" in list(model_cursor.keys()):
                label = model_cursor["label"]
                print("  label : {}".format(label))
                break
            else:
                attribute = model_cursor["attribute"]
                value = input_data[attribute]
                model_cursor = next(
                    category for category in model_cursor["categories"] \
                        if category["value"] == value)["tree"]
                print("  {} : {}".format(attribute, value))
            i_value += 1
        return label

    def fit(self, train_data):
        """Build the tree.
        Rules of recursion:
            1) Believe that it works.
            2) Start by checking for the base case (no further information gain).
            3) Prepare forgiant stack traces.
        Args:
            train_data (list): inputs data.
        """

        # N = len(data_df)
        output_attribute = list(data_df.columns)[-1]
        # Check if all records have the same output value
        output_serie = data_df[output_attribute].value_counts()
        if len(output_serie) == 1:
            label = pd.Index(output_serie).get_loc(len(data_df))
            self.model["label"] = label
        else:
            splitting_attribute = self.compute_best_attribute(data_df)
            self.model["attribute"] = splitting_attribute
            self.model["categories"] = []
            for value in data_df[splitting_attribute].unique():
                self.model["categories"].append(
                    {
                        "value": value,
                        "tree": {}
                    }
                )

    def compute_best_attribute(data_df):
        """Compute the best attribute."""


    def measure_impurity(self, y_array, method="entropy"):
        """
        Compute the information in a given set post spit.
        Args:
            y_array = [0, 1, 1, ...]
        """
        unique_values = list(set(y_array))
        if method == "gini_impurity":
            print("to be coded")
        elif method == "entropy":
            probabilities = []
            for value in unique_values:
                prob = y_array.count(value)
                probabilities.append(-prob*np.log2(prob))
            impurity = np.sum(probabilities)
        return impurity

