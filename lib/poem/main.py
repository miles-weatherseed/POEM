import argparse
import numpy as np
import pandas as pd
import os
import yaml
from typing import Dict, Any
from poem.utils.utils import make_poem_input_layer, VALID_ENCODINGS
from sklearn.preprocessing import StandardScaler, MinMaxScaler

current_dir = os.path.dirname(os.path.abspath(__file__))


# function to convert command line inputs into lower case
def case_insensitive_mode(value):
    return value.casefold()


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
        type=case_insensitive_mode,  # Convert to lowercase (case-insensitive)
        choices=["train", "test"],  # Restricted choices
    )

    parser.add_argument(
        "-c",
        "--config_path",
        help="Path to config file.",
        type=str,
        default=os.path.join(current_dir, "poem_settings.yml"),
    )
    # Parse the arguments
    args = parser.parse_args()
    config = yaml.safe_load(
        open(
            args.config_path,
            "r",
        )
    )
    # check config will work
    config = validate_yaml_string_inputs(config)
    config = validate_yaml_numerical_inputs(config)

    # if this is a valid yaml then we can move on to train/test
    if args.mode == "train":
        train_poem(config)
    elif args.mode == "test":
        test_poem(config)

    return


def train_poem(training_data: pd.DataFrame, config: Dict[str, Any]):
    # We re-train POEM using mechanistic output and user-defined settings
    # Create input data
    poem_input = make_poem_input_layer(
        training_data["mhc_i"].values,
        training_data["length"].values,
        training_data["peptide"].values,
        training_data["predicted_pmhc"].values,
        config["dataset"]["length_encoding"],
        config["dataset"]["mhci_encoding"],
        config["dataset"]["mhci_sequence"],
        config["dataset"]["tcr_encoding"],
    )

    # create a pipeline to hold the scaler and the regressor
    return


def test_poem(test_data: pd.DataFrame, config: Dict[str, Any]):
    poem_input = make_poem_input_layer(
        test_data["mhc_i"].values,
        test_data["length"].values,
        test_data["peptide"].values,
        test_data["predicted_pmhc"].values,
        config["dataset"]["length_encoding"],
        config["dataset"]["mhci_encoding"],
        config["dataset"]["mhci_sequence"],
        config["dataset"]["tcr_encoding"],
    )

    # load model

    # make predictions
    return


# Functions to check user settings will work
PERMITTED_TCR_ENCODINGS = ["nettepi", "prime", "none"]
PERMITTED_MHCI_SEQUENCE = ["pseudosequence", "full", "none"]
PERMITTED_MHCI_ENCODINGS = [enc.casefold() for enc in VALID_ENCODINGS] + [
    "none]"
]
PERMITTED_NORMALIZATIONS = ["minmax", "standard", "none"]
# TODO: add other tensorflow activation functions to this list
PERMITTED_ACTIVATIONS = ["relu", "sigmoid", "tanh"]
PERMITTED_OPTIMIZERS = ["sgd", "adam"]
PERMITTED_LOSS_FUNCTIONS = ["binary_crossentropy"]


def validate_yaml_string_inputs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes the YAML to ensure free text user-provided
    options will work.

    Args:
        config (Dict[str, Any]): Loaded config from YAML

    Returns:
        Dict[str, Any]: Processed and validated YAML
    """

    # check dataset options
    for opt, perm in zip(
        ["tcr_encoding", "mhci_sequence", "mhci_encoding", "normalization"],
        [
            PERMITTED_TCR_ENCODINGS,
            PERMITTED_MHCI_SEQUENCE,
            PERMITTED_MHCI_ENCODINGS,
            PERMITTED_NORMALIZATIONS,
        ],
    ):
        # convert to lower case
        config["dataset"][opt] = config["dataset"][opt].casefold()
        # check in permitted list
        if config["dataset"][opt] not in perm:
            raise ValueError(
                f"Invalid choice of {opt}: {config['dataset'][opt]}. "
                f"Must be one of: {', '.join(perm)}"
            )

    # check model options
    for opt, perm in zip(
        ["activation", "output_activation"],
        [
            PERMITTED_ACTIVATIONS,
            PERMITTED_ACTIVATIONS,
        ],
    ):
        # convert to lower case
        config["model"][opt] = config["model"][opt].casefold()
        # check in permitted list
        if config["model"][opt] not in perm:
            raise ValueError(
                f"Invalid choice of {opt}: {config['model'][opt]}. "
                f"Must be one of: {', '.join(perm)}"
            )

    # check training options
    for opt, perm in zip(
        ["optimizer", "loss_function"],
        [
            PERMITTED_OPTIMIZERS,
            PERMITTED_LOSS_FUNCTIONS,
        ],
    ):
        # convert to lower case
        config["training"][opt] = config["training"][opt].casefold()
        # check in permitted list
        if config["training"][opt] not in perm:
            raise ValueError(
                f"Invalid choice of {opt}: {config['training'][opt]}. "
                f"Must be one of: {', '.join(perm)}"
            )

    return config


def validate_yaml_numerical_inputs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validates the YAML to ensure numerical user-provided options will work.

    Args:
        config (Dict[str, Any]): Loaded config from YAML

    Returns:
        Dict[str, Any]: Processed and validated YAML
    """
    # check dropout_rate is a float
    if not isinstance(config["model"]["dropout_rate"], float):
        raise ValueError(
            f"Invalid dropout_rate: {config['model']['dropout_rate']}. "
            "Must be a float."
        )
    # check dropout_rate is in [0.0, 1.0)
    if not (0.0 <= config["model"]["dropout_rate"] < 1.0):
        raise ValueError(
            f"Invalid dropout_rate: {config['model']['dropout_rate']}. "
            "Must be in [0.0, 1.0)"
        )
    # check hidden_layers is either int or a list of ints
    if not isinstance(config["model"]["hidden_layers"], int):
        if not isinstance(config["model"]["hidden_layers"], list):
            raise ValueError(
                f"Invalid hidden_layers: {config['model']['hidden_layers']}. "
                "Must be an int or a list of ints."
            )
        else:
            # still need to check values are ints
            if not all(
                isinstance(i, int) for i in config["model"]["hidden_layers"]
            ):
                raise ValueError(
                    f"Invalid hidden_layers: {config['model']['hidden_layers']}. "
                    "Must be a list of integers."
                )
            else:
                # save number of hidden layers for ease later
                config["model"]["n_hidden_layers"] = len(
                    config["model"]["hidden_layers"]
                )
    # check validation_split is a float
    if not isinstance(config["training"]["validation_split"], float):
        raise ValueError(
            f"Invalid validation_split: {config['training']['validation_split']}. "
            "Must be a float."
        )
    # check validation_split is in [0.0, 1.0)
    if not (0.0 <= config["training"]["validation_split"] < 1.0):
        raise ValueError(
            f"Invalid validation_split: {config['training']['validation_split']}. "
            "Must be in [0.0, 1.0)"
        )
    # check epochs is an int
    if not isinstance(config["training"]["epochs"], int):
        raise ValueError(
            f"Invalid epochs: {config['training']['epochs']}. "
            "Must be an int."
        )
    # check batch_size is an int
    if not isinstance(config["training"]["batch_size"], int):
        raise ValueError(
            f"Invalid batch_size: {config['training']['batch_size']}. "
            "Must be an int."
        )
    # check patience is an int
    if not isinstance(config["training"]["early_stopping"]["patience"], int):
        raise ValueError(
            f"Invalid early stopping patience: {config['training']['early_stopping']['patience']}. "
            "Must be an int."
        )
    # check learning_rate is a float
    if not isinstance(config["training"]["learning_rate"], float):
        raise ValueError(
            f"Invalid learning_rate: {config['training']['learning_rate']}. "
            "Must be a float."
        )
    return config
