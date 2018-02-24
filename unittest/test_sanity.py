"""Sanity test script.

This module tests the basic components of the module.
"""
import unittest

import prepare
import train
import predict

MODELS = ["python_CART", "python_ID3"]

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
        for model in MODELS:
            train.main(train_data=train_data,
                       dataset_name=None,
                       model_type=model,
                       start_version=None,
                       end_version="unittest")

        # Retrieving the data
        for dataset in ["us_election"]:
            for model in MODELS:
                train.main(train_data=None,
                           dataset_name=dataset,
                           model_type=model,
                           start_version=None,
                           end_version="unittest")


    def test_predict(self):
        """Test the prepare module.
        """
        for model in MODELS:
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
