"""Sanity test script.

This module tests the basic components of the module.
"""
import unittest

import prepare
import train
import predict
import version
from context.vanilla import vanilla


class TestSanity(unittest.TestCase):
    """Class to test the core module.

    """
    def test_prepare(self):
        """Test the prepare module.
        """
        for dataset in ["dataset_1_name"]:
            prepare.main(dataset)

    def test_train(self):
        """Test the prepare module.
        """
        train_data = []
        # Inputing the data
        for model in ["diy", "scikit_learn"]:
            train.main(train_data=train_data,
                       dataset_name=None,
                       model_type=model,
                       start_version=None,
                       end_version="unittest")

        # Retrieving the data
        for dataset in ["us_election"]:
            for model in ["diy", "scikit_learn"]:
                train.main(train_data=None,
                           dataset_name=dataset,
                           model_type=model,
                           start_version=None,
                           end_version="unittest")


    def test_predict(self):
        """Test the prepare module.
        """
        for model in ["diy", "scikit_learn"]:
            # Train
            train.main(train_data=None,
                       dataset_name="us_election",
                       model_type=model,
                       start_version=None,
                       end_version="unittest")
            # Inputing the data
            predict.main(inputs_data=[],
                         model_type=model,
                         model_version="us_election",
                         model=None)

    # def test_vanilla(self):
    #     """Test the functionnalities of vanilla layer context.
    #     """
    #     # Input for train
    #     train_input_fields = {
    #         "train_data": [
    #             {
    #                 "id_data": "12dc",
    #                 "input": [1, 2, 3],
    #                 "output": [4]
    #             }
    #         ]
    #     }
    #     predict_input_fields = {
    #         "predict_data": [
    #             {
    #                 "id_data": "string",
    #                 "input": [1, 2, 3]
    #             }
    #         ]
    #     }
    #     # For each model
    #     for model in ["sum_plus_x", "sum_minus_x"]:
    #         # Train from scratch
    #         vanilla.train_api(model_type=model,
    #                           model_version="unittest",
    #                           train_input_fields=train_input_fields)
    #         # Update
    #         vanilla.update_api(model_type=model,
    #                            start_version="unittest",
    #                            end_version="unittest",
    #                            train_input_fields=train_input_fields)
    #         # Predict
    #         vanilla.predict_api(model_type=model,
    #                             model_version="unittest",
    #                             predict_input_fields=predict_input_fields)
    #         # Read
    #         vanilla.read_api(model_type=model,
    #                          model_version="unittest")
    #         # Delete
    #         vanilla.delete_api(model_type=model,
    #                            model_version="unittest")
