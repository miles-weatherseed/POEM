import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import joblib
import os
from mechanistic_model.utils.utils import (
    encode_tap_peptides,
    generate_tap_training_data,
)

current_dir = os.path.dirname(os.path.abspath(__file__))

PAD_START = 3
PAD_LENGTH = 11
ensemble_encodings = [
    "sparse",
    "blosum45",
    "levin",
    "rao",
    "risler",
    "physico",
]
Cs = [
    9.722712197452008,
    15.930522616241015,
    11.713658273538,
    31.7498834642438,
    81.9599661292727,
    15.930522616241,
]
epsilons = [
    0.003439284699685042,
    0.13311216080736885,
    0.177718340164964,
    0.183496853769404,
    0.107228372229983,
    0.133112160807368,
]


def append_sparse_length(data, lengths):
    """
    Takes in dataset (n, m) and lengths (n,) and creates a sparse encoding of the lengths 8 --> 16, i.e. (n, 9) and sticks this on the side
    """
    n = data.shape[0]
    sparse_out = np.zeros((n, 8))
    for i, l in enumerate(lengths):
        sparse_out[i, min(l - 8, 7)] = 1
    return np.hstack([data, sparse_out])


def train_models(
    encodings: list[str], Cs: list[float], epsilons: list[float]
) -> list:
    """Re-trains models and groups together so that we don't run into any sklearn version issues

    Args:
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
                        kernel="rbf",
                        C=Cs[i],
                        epsilon=epsilons[i],
                        cache_size=1600,
                        max_iter=1000000000,
                    ),
                ),
            ]
        )

        X, y, train_length = generate_tap_training_data(
            PAD_START,
            enc,
            PAD_LENGTH,
        )
        X = append_sparse_length(X, train_length)
        pipeline.fit(X, y)
        models.append(pipeline)
    return models


def generate_ensemble_models():
    """
    This function should be called when POEM is installed on a new machine,
    creating new versions of the regressors and avoiding and scikit-learn
    compatability issues
    """
    trained_models = train_models(ensemble_encodings, Cs, epsilons)
    for enc, model in zip(ensemble_encodings, trained_models):
        joblib.dump(
            model,
            open(
                os.path.join(current_dir, "cache", f"tap_models/{enc}.reg"),
                "wb",
            ),
            protocol=5,
        )


# if cache of ensemble models doesn't exist, we should re-train it
if any(
    [
        not os.path.exists(
            os.path.join(current_dir, "cache", f"tap_models/{enc}.reg")
        )
        for enc in ensemble_encodings
    ]
):
    generate_ensemble_models()

models = [
    joblib.load(
        open(os.path.join(current_dir, "cache", f"tap_models/{enc}.reg"), "rb")
    )
    for enc in ensemble_encodings
]


def predict_tap_binding_affinities(
    peptides: np.ndarray, host: str
) -> np.ndarray:
    """Function takes in array of peptide sequences and an array of host
    organisms. Returns the predicted TAP binding affinity (nM).

    The function can handle peptides of any length >= 8. However, the model
    is only trained using lengths up to and including 15 amino acids.

    Args:
        peptides (np.ndarray): Array of peptide sequences (lengths >= 8)
        host (str): Host organism

    Returns:
        np.ndarray: predicted TAP IC50 (nM)
    """
    lengths = np.array([len(p) for p in peptides])
    preds = np.zeros((len(peptides), len(ensemble_encodings)))
    hosts = np.array([host for _ in range(len(peptides))])
    for j, enc in enumerate(ensemble_encodings):
        model = models[j]
        X = encode_tap_peptides(peptides, hosts, PAD_START, enc, PAD_LENGTH)
        X = append_sparse_length(X, lengths)
        preds[:, j] = model.predict(X)
    # return arithmetic mean across ensemble, raised to power of 10 to reverse
    # log scaling
    return 10 ** np.mean(preds, axis=1)


if __name__ == "__main__":
    print(
        predict_tap_binding_affinities(
            np.array(["QLESIINFEKL", "FSIINFEKL"]),
            np.array(["Human", "Mouse"]),
        )
    )
