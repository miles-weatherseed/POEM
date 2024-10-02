import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from utils.utils import (
    make_length,
    sparse_peptide_encoding,
    VALID_ENCODINGS,
    physico_peptide_encoding,
    submatrix_peptide_encoding,
)
import joblib
import os

PAD_START = 5
PAD_LENGTH = 11

# slope to convert regressor output to kcats - from M_Weatherseed PhD thesis.
SLOPE = 0.0400

# best performing encodings and corresponding hyperparameters from M_Weatherseed PhD thesis
encodings = ["benner74", "blosum50", "feng", "gonnet1992", "levin"]
Cs = [
    1.0669882710369707,
    6.935400690096964,
    96.8120891912843,
    1.386378561590795,
    58.47739859832214,
]
epsilons = [
    0.0039493644472223316,
    0.22594607000759603,
    0.0016649059071639674,
    0.015500706089539403,
    0.17504766488065068,
]
kernels = ["linear", "linear", "rbf", "linear", "rbf"]


def generate_datasets_erap(
    peptides: np.ndarray, startpoint: int, encoding: str, length: int
) -> np.ndarray:
    """_summary_

    Generates the datasets required for training the ERAP1 predictor, based on
    user desired encoding and padding strategies.
    Args:
        peptides (np.ndarray): Array of peptide strings in the training data.
        startpoint (int): Which point in the peptide to start padding or trimming (N.B. corresponds to P_{i+1} due to Python's indexing on 0)
        encoding (str): Which encoding method to use (e.g. "BLOSUM62")
        length (int): The length to pad/trim all peptides to

    Returns:
        np.ndarray: array of encoded and length-normalised peptide representations.
    """

    normalised_peptides = make_length(peptides, startpoint, length)

    if encoding not in VALID_ENCODINGS:
        raise ValueError("Must choose a method in VALID_ENCODINGS")

    if encoding == "sparse":
        train_peptide_encodings = sparse_peptide_encoding(normalised_peptides)
    elif encoding == "physico":
        train_peptide_encodings = physico_peptide_encoding(normalised_peptides)
    else:
        train_peptide_encodings = submatrix_peptide_encoding(
            normalised_peptides, encoding
        )
    return train_peptide_encodings


def generate_ensemble_models():
    """
    This function should be called when POEM is installed on a new machine,
    creating new versions of the regressors and avoiding and scikit-learn
    compatability issues
    """
    # load training data
    data = pd.read_csv("lib/mechanistic_model/data/erap1_training_data.csv")
    # count number of times each peptide appears in data
    data["count"] = [
        sum(data.peptide.values == p) for p in data.peptide.values
    ]
    # we weight the training data inversely by number of appearances of each peptide
    weights = 1 / data["count"].values
    # we log scale and truncate at ~1% trimming rate per hour
    y = np.log10(data.trimming_rate.values)
    y[y <= -0.5] = -0.5
    peptides = data.peptide.values
    # create ensemble
    trained_models = train_models(
        weights, peptides, y, encodings, Cs, epsilons
    )
    # save these into a cache for easy loading later
    for enc, model in zip(encodings, trained_models):
        joblib.dump(
            model,
            open(f"lib/mechanistic_model/cache/erap1_models/{enc}.reg", "wb"),
            protocol=5,
        )
    return


def train_models(
    sample_weights: np.ndarray,
    peptides: np.ndarray,
    y: np.ndarray,
    encodings: list[str],
    Cs: list[float],
    epsilons: list[float],
) -> list:
    """Re-trains models and groups together so that we don't run into any sklearn version issues

    Args:
        sample_weights (np.ndarray): 1/peptide_count, where peptide_count is the number of times the peptide is repeated in the training data
        peptides (np.ndarray): _description_
        y (np.ndarray): log10 of the trimming rate
        encodings (list[str]): list of amino acid encodings in the ensemble
        Cs (list[float]): list of SVR regularisation terms in the ensemble
        epsilons (list[float]): list of epsilons for softmargin lost function

    Returns:
        list: list of trained regressors in ensemble
    """
    models = []

    for i, enc in enumerate(encodings):
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                (
                    "regressor",
                    SVR(
                        kernel=kernels[i],
                        C=Cs[i],
                        epsilon=epsilons[i],
                        cache_size=1600,
                        max_iter=1000000000,
                    ),
                ),
            ]
        )
        X = generate_datasets_erap(
            peptides,
            startpoint=PAD_START,
            encoding=enc,
            length=PAD_LENGTH,
        )
        pipeline.fit(X, y, regressor__sample_weight=sample_weights)
        models.append(pipeline)

    return models


# if cache of ensemble models doesn't exist, we should re-train it
if any(
    [
        not os.path.exists(
            f"lib/mechanistic_model/cache/erap1_models/{enc}.reg"
        )
        for enc in encodings
    ]
):
    generate_ensemble_models()

models = [
    joblib.load(
        open(f"lib/mechanistic_model/cache/erap1_models/{enc}.reg", "rb")
    )
    for enc in encodings
]


def predict_scores(peptides: np.ndarray) -> np.ndarray:
    """Runs each regressor in ensemble and returns arithmetic mean of their scores

    Args:
        peptides (np.ndarray): Array of peptides to form predictions for

    Returns:
        np.ndarray: A score for each peptide, corresponding to the expected
        rate of trimming by ERAP1 in specific assay conditions (defined in M_Weatherseed thesis)
    """

    predictions = np.zeros((len(peptides), len(encodings)))
    for i in range(len(encodings)):
        X = generate_datasets_erap(
            peptides,
            PAD_START,
            encodings[i],
            PAD_LENGTH,
        )
        predictions[:, i] = models[i].predict(X)

    return np.mean(predictions, axis=1)


def get_erap1_kcats(peptides: np.ndarray) -> np.ndarray:
    """Takes in array of peptides and returns an array of the predicted kcats (in s^-1)

    Args:
        peptides (np.ndarray): Peptide sequences to make predictions on

    Returns:
        np.ndarray: predicted kcats (in s^-1)
    """
    return SLOPE * 10 ** (predict_scores(peptides))


if __name__ == "__main__":
    # write this into tests eventually - check you get known output
    print(get_erap1_kcats(np.array(["SIINFEKLM", "SIINYEKL"])))
