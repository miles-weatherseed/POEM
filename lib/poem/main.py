import argparse
import numpy as np
import pandas as pd
import os
import yaml
from poem.utils.utils import encode_nettepi_tcr_residues

current_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Provide data to train or test POEM."
    )

    # Adding -p for input_peptides (CSV file)
    parser.add_argument(
        "-d",
        "--input_data",
        help="Path to the input data CSV file.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-m",
        "--mode",
        help="Whether to train a new model or test one.",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-c",
        "--config",
        help="Path to config file.",
        type=str,
        default=os.path.join(current_dir, "poem_settings.yml"),
    )
    # Parse the arguments
    args = parser.parse_args()
    return


def train_poem():
    return


def test_poem():
    return
