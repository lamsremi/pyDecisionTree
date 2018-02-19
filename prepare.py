"""Script to prepare the data.
"""
import importlib


def main(data_source):
    """Prepare the data.

    Args:
        data_source (str): source of the data to prepare.
    """
    # Load preprocess file
    preprocess_module = importlib.import_module("data.{}.preprocess".format(data_source))

    # Main
    preprocess_module.main()

if __name__ == '__main__':
    for source in ["us_election"]:
        main(data_source=source)
