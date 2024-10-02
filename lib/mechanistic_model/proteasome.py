import subprocess
import numpy as np
from typing import Optional

# scaling parameters from M_Weatherseed PhD research
param_dict = {
    0: np.array([0.57772718, 0.19649221]),
    1: np.array([0.3428299, 0.19196815]),
    2: np.array([0.31844389, 0.18393788]),
}


def run_pepsickle(fasta_file: str, *args) -> list:
    """Runs pepsickle and captures the output for usage

    Args:
        fasta_file (str): Path to fasta file on which to run Pepsickle algorithm

    Returns:
        list: List of line-by-line output
    """
    # Run pepsickle and capture the output
    command = [
        "pepsickle",
        "-f",
        fasta_file,
        *args,
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode == 0:
        # Process the output as needed
        output_lines = result.stdout.split("\n")
        return [x.split("\t") for x in output_lines]

    else:
        # Handle the case where the command failed
        print(f"Error running pepsickle. Return code: {result.returncode}")
        return None


def run_algos(
    fasta_filepath: str, model_idx: int, pepsickle_version: Optional[str]
) -> np.ndarray:
    """Takes in filepath and model index. Runs algo and returns probabilities
    vector, \hat{p}

    Args:
        fasta_filepath (str): _description_
        model_idx (int): Which Pepsickle model to use ([0] Pepsickle epitope;
        [1] Pepsickle in-vitro; [2] Pepsickle in-vitro-2)
        pepsickle_version (Optional[str]): Pepsickle version: constitutive (C)
        or immunoproteasome (I)

    Returns:
        np.ndarray: array of residue cleavage scores for all internal residues
        (C-term of P0 to PN-1)
    """

    pepsickle_model = ["epitope", "in-vitro", "in-vitro-2"][model_idx]
    additional_args = [
        "-p",
        pepsickle_version,
        r"-m",
        rf"{pepsickle_model}",
    ]
    pepsickle_output = run_pepsickle(fasta_filepath, *additional_args)
    scores = np.array([float(x[2]) for x in pepsickle_output[1:-1]])
    return scores


def calculate_Pij(
    start: int,
    end: int,
    p: np.ndarray,
    p_hat: np.ndarray,
    r: float,
) -> float:
    """Returns the probability of proteasome forming the peptide spanning from
    residues i (start) to j (end) inclusive

    Args:
        start (int): Index of start of peptide being formed from the protein
        end (int): Index of end of peptide being formed from the protein
        p (np.ndarray): Vector of probability of cleavage when previous
        cleavage position is unknown
        p_hat (np.ndarray): Vector of probability of cleavage when previous
        cleavage is more than 3 residues away
        r (float): Proportion reduction in p_hat when previous cleavage was
        less than or equal to 3 residues away

    Returns:
        float: Probability of peptide forming from a single protein
    """
    l = 3
    assert end - start > 0  # otherwise this is nonsense!
    # we make case distinctions based on how far end is from start
    if end - start < l:
        pij = (
            p[start]
            * np.prod(1 - r * p_hat[start + 1 : end + 1])
            * r
            * p_hat[end + 1]
        )
    elif end - start == l:
        pij = (
            p[start]
            * np.prod(1 - r * p_hat[start + 1 : end + 1])
            * p_hat[end + 1]
        )
    else:
        pij = (
            p[start]
            * np.prod(1 - r * p_hat[start + 1 : start + 1 + l])
            * np.prod(1 - p_hat[start + l + 1 : end + 1])
            * p_hat[end + 1]
        )
    return pij


def get_proteasome_product_probabilities(
    fasta_filepath: str,
    model_idx: int,
    startpoints: np.ndarray,
    lengths: np.ndarray,
    direction: Optional[str] = "both",
    pepsickle_version: Optional[str] = "I",
) -> np.ndarray:
    """Takes in a fasta filepath, runs Pepsickle and returns probability of
    peptides being formed in a numpy array.

    N.B. we replace the cleavage probability of the C-terminus with 1.

    Models:

    [0] Pepsickle epitope
    [1] Pepsickle in-vitro
    [2] Pepsickle in-vitro-2

    Args:
        fasta_filepath (str): Protein fasta filepath
        model_idx (int): Index of Pepsickle model to use
        startpoints (np.ndarray): Indexes of starts of peptides being formed from the protein
        lengths (np.ndarray): Lengths of peptides being formed from the protein
        direction (Optional[str], optional): Direction of proteasomal digestion. Defaults to "both".
        pepsickle_version (Optional[str], optional): Whether to use constitutive ("C") or immunoproteasome ("I"). Defaults to "I".

    Returns:
        np.ndarray: Array of peptide formation probabilities
    """
    # calculates p from p_hat
    gamma, r = param_dict[model_idx]
    l = 3
    assert len(startpoints) == len(lengths)
    output = np.zeros(len(startpoints))
    if direction == "N":
        p_hat = np.hstack(
            [
                1.0,
                gamma
                * run_algos(fasta_filepath, model_idx, pepsickle_version)[:-1],
                1.0,
            ]
        )

        # hold pX probabilities from 1 to l inclusive
        pX = np.zeros((len(p_hat), l))
        p = np.zeros_like(p_hat)
        pX[0, 0] = 0  # X_{-1} undefined so set to 0
        p[0] = 1
        for i in range(1, len(p) - 1):
            pX[i, 0] = p[i - 1]
            for j in range(1, l):
                pX[i, j] = pX[i - 1, j - 1] * (1 - r * p_hat[i - 1])
            p[i] = p_hat[i] * (1 - np.sum(pX[i, :])) + r * p_hat[i] * np.sum(
                pX[i, :]
            )
        p[-1] = 1  # C-terminus already cleaved

        for i, (s, length) in enumerate(zip(startpoints, lengths)):
            output[i] = calculate_Pij(s, s + length - 1, p, p_hat, r)

    elif direction == "C":
        p_hat = np.hstack(
            [
                1.0,
                gamma
                * run_algos(fasta_filepath, model_idx, pepsickle_version)[
                    :-1:-1
                ],
                1.0,
            ]
        )

        # hold pX probabilities from 1 to l inclusive
        pX = np.zeros((len(p_hat), l))
        p = np.zeros_like(p_hat)
        pX[0, 0] = 0  # X_{-1} undefined so set to 0
        p[0] = 1
        for i in range(1, len(p) - 1):
            pX[i, 0] = p[i - 1]
            for j in range(1, l):
                pX[i, j] = pX[i - 1, j - 1] * (1 - r * p_hat[i - 1])
            p[i] = p_hat[i] * (1 - np.sum(pX[i, :])) + r * p_hat[i] * np.sum(
                pX[i, :]
            )
        p[-1] = 1  # C-terminus already cleaved

        for i, (s, length) in enumerate(zip(startpoints, lengths)):
            output[i] = calculate_Pij(
                len(p) - s - length - 1, len(p) - 2 - s, p, p_hat, r
            )

    elif direction == "both":
        # have to replace the last score with a 1 because C-terminus is already cleaved
        algo_out = run_algos(fasta_filepath, model_idx, pepsickle_version)[:-1]
        p_hat_N = np.hstack(
            [
                1.0,
                gamma * algo_out,
                1.0,
            ]
        )
        p_hat_C = np.hstack(
            [
                1.0,
                gamma * algo_out[::-1],
                1.0,
            ]
        )

        # hold pX probabilities from 1 to l inclusive
        pX_N = np.zeros((len(p_hat_N), l))
        pX_C = np.zeros((len(p_hat_C), l))
        p_N = np.zeros_like(p_hat_N)
        p_C = np.zeros_like(p_hat_C)
        pX_N[0, 0] = 0  # X_{-1} undefined so set to 0
        pX_C[0, 0] = 0  # X_{-1} undefined so set to 0
        p_N[0] = 1
        p_C[0] = 1
        for i in range(1, len(p_N) - 1):
            pX_N[i, 0] = p_N[i - 1]
            pX_C[i, 0] = p_C[i - 1]
            for j in range(1, l):
                pX_N[i, j] = pX_N[i - 1, j - 1] * (1 - r * p_hat_N[i - 1])
                pX_C[i, j] = pX_C[i - 1, j - 1] * (1 - r * p_hat_C[i - 1])
            p_N[i] = p_hat_N[i] * (1 - np.sum(pX_N[i, :])) + r * p_hat_N[
                i
            ] * np.sum(pX_N[i, :])
            p_C[i] = p_hat_C[i] * (1 - np.sum(pX_C[i, :])) + r * p_hat_C[
                i
            ] * np.sum(pX_C[i, :])
        p_N[-1] = 1  # C-terminus already cleaved
        p_C[-1] = 1  # C-terminus already cleaved

        for i, (s, length) in enumerate(zip(startpoints, lengths)):
            output[i] = 0.5 * (
                calculate_Pij(s, s + length - 1, p_N, p_hat_N, r)
                + calculate_Pij(
                    len(p_C) - s - length - 1,
                    len(p_C) - 2 - s,
                    p_C,
                    p_hat_C,
                    r,
                )
            )

    else:
        raise ValueError("Direction must be one of 'N', 'C', or 'both'")
    return output


if __name__ == "__main__":
    fasta_filepath = "test/test.fasta"
    print(run_algos(fasta_filepath, model_idx=2, pepsickle_version="I"))
    print(
        get_proteasome_product_probabilities(
            fasta_filepath,
            model_idx=2,
            startpoints=[10],
            lengths=[8],
        )
    )
    print("SLEQLESIINFEKLGLEM"[10 : 10 + 8])
