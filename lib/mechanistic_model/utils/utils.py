import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
import json
import os

# Get the absolute path to the directory where this script resides
current_dir = os.path.dirname(os.path.abspath(__file__))

# load substitution matrices, define AAs and load in physicochemical features
SUBS_MATRICES = substitution_matrices.load()
AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",
]  # canonical amino acids
PHYSICO_DICT = json.load(
    open(os.path.join(current_dir, "aaindex1_pca.json"), "r")
)
VALID_ENCODINGS = [x.lower() for x in SUBS_MATRICES] + [
    "physico",
    "sparse",
]
HUMAN_TAP_PSEUDOSEQ = "SSSERCTMNPLVRARMTI"
MOUSE_TAP_PSEUDOSEQ = "CSVQSSTMRPFIKDRINM"
RAT_A_TAP_PSEUDOSEQ = "CNVESSAEQSLIKSQINM"
RAT_U_TAP_PSEUDOSEQ = "CNVESSTMRPFIKERINM"

# PEPTIDE HANDLING FUNCTIONS


# Functions facilitating general manipulation of peptide sequences to vectors


def sparse_peptide_encoding(peptides: list) -> np.ndarray:
    """_summary_
    Takes a peptide list input and returns a sparse representation of the peptides.
    Args:
        peptide (list): Peptides to be encoded
    """
    lengths = set([len(x) for x in peptides])
    assert len(lengths) == 1  # can't handle mixed lengths
    output = np.zeros((len(peptides), 21 * len(peptides[0])))
    for i in range(len(peptides)):
        for idx, s in enumerate(peptides[i]):
            output[i, idx * 21 + AMINO_ACIDS.index(s)] += 1
    return np.array(output)


def physico_peptide_encoding(peptides: list) -> np.ndarray:
    """_summary_

    Args:
        peptides (list): List of peptides to encode using length 19 PCA decomposition of AAIndex features
    """
    outputs = np.zeros((len(peptides), len(peptides[0]) * 19))
    for i in range(len(peptides)):
        outputs[i, :] = np.array(
            [PHYSICO_DICT[s] for s in peptides[i]]
        ).flatten()
    return outputs


def submatrix_peptide_encoding(peptides: list, matrix: str) -> np.ndarray:
    """_summary_
    Converts a list of peptides to substitution matrix encodings.
    Args:
        peptide (list): List of peptides to be encoded
        matrix (str): The substitution matrix name to use (e.g. "BLOSUM62")
    """
    assert matrix.upper() in SUBS_MATRICES
    M = substitution_matrices.load(name=matrix.upper())

    # some of the substitution matrices in the Biopython package don't have the X amino acid
    # TODO: decide on better way to handle these. Replace with alanine for now.

    if ("X", "X") not in M.keys():
        for i in range(len(peptides)):
            peptides[i] = peptides[i].replace(
                "X", "A"
            )  # switch to alanine padding for now
    return np.array(
        [np.array([M[x].values() for x in p]).flatten() for p in peptides]
    )


def pad_peptide_tolength(peptide: str, pad_start: int, length: int) -> str:
    """
    This function takes in a peptide and pads with the unknown amino acid
    X, starting at site pad_start.
    """
    return (
        peptide[:pad_start]
        + "X" * (length - len(peptide))
        + peptide[pad_start:]
    )


def trim_peptide_tolength(peptide: str, trim_start: int, length: int) -> str:
    """
    This function takes in a list of peptides of different lengths and
    converts it to a list of nmers by trimming internal amino acids, starting
    at site trim_start.
    """
    return peptide[:trim_start] + peptide[-(length - trim_start) :]


def make_length(peptides: np.ndarray, start: int, length: int) -> list:
    """Takes in peptides and converts them all to the same length by a mixture
    of padding and trimming

    Args:
        peptides (np.ndarray): Array of peptides to be normalised.
        start (int): Point at which to start trimming or padding.
        length (int): The length to normalise all peptides to.

    Returns:
        list: _description_
    """
    output = list(peptides)
    output = [
        pad_peptide_tolength(x, start, length) if len(x) < length else x
        for x in output
    ]
    output = [
        trim_peptide_tolength(x, start, length) if len(x) > length else x
        for x in output
    ]
    return output


# FUNCTIONS TO HANDLE LARGE PROTEINS


def find_peptide_precursors(
    length: int, fasta_sequence: str, lengths: list, start_position: int
):
    end_position = start_position + length - 1
    peptides = [
        "A" * max(-(end_position - x + 1), 0)
        + fasta_sequence[max(end_position - x + 1, 0) : end_position + 1]
        for x in lengths
    ]  # pre-pad with alanine if needed - the proteasome model will capture that this doesn't exist
    return peptides


def protein_to_peptides(protein_sequence: str):
    # this returns a list of startpoint and list of endpoints of the peptides we must keep track of, in order
    N = len(protein_sequence)
    startpoints = []
    endpoints = []
    # initial peptides
    for i in range(7, 15):
        for j in range(i - 7 + 1):
            startpoints.append(j)
            endpoints.append(i)
    for i in range(15, N):
        for j in range(i - 15, i - 6):
            startpoints.append(j)
            endpoints.append(i)

    lengths = [e - s + 1 for s, e in zip(startpoints, endpoints)]

    return startpoints, endpoints, lengths


def out_to_in(out_rates: np.ndarray):
    # This function will take catalytic rates corresponding to ERAP1 and cytosolic aminopeptidases for the peptides in the system
    # It will return a new array containing these rates rearranged to so that the precursor is in the corresponding index
    in_rates = np.zeros_like(out_rates)
    N = int(len(out_rates) / 9 + 11)
    for i in range(7, 15):
        start = int(0.5 * (i - 7) * (i - 6))
        for j in range(1, i - 7 + 1):
            in_rates[start + j] = out_rates[start + j - 1]
    for i in range(15, N):
        start = 36 + (i - 15) * 9
        for j in range(1, 9):
            in_rates[start + j] = out_rates[start + j - 1]
    return in_rates


def encode_tap_allele(alleles: np.ndarray) -> list:
    allele_list = []
    for allele in alleles:
        if allele == "Human":
            allele_list.append(HUMAN_TAP_PSEUDOSEQ)
        elif allele == "Mouse":
            allele_list.append(MOUSE_TAP_PSEUDOSEQ)
        elif allele == "Rat_a":
            allele_list.append(RAT_A_TAP_PSEUDOSEQ)
        elif allele == "Rat_u":
            allele_list.append(RAT_U_TAP_PSEUDOSEQ)
    return allele_list


def generate_tap_training_data(
    startpoint: int,
    encoding: str,
    length: int,
):
    """_summary_

    Args:
        method (str): Whether to pad or to trim in order to get all peptides to the same length
        startpoint (int): Which point in the peptide to start padding or trimming (N.B. corresponds to P_{i+1})
        encoding (str): Which encoding method to use
    """
    train_data = pd.read_csv(
        os.path.abspath(
            os.path.join(current_dir, "..", "data", "tap_training_data.csv")
        ),
    )

    train_peptides = make_length(train_data.Peptide.values, startpoint, length)
    if encoding not in VALID_ENCODINGS:
        raise ValueError("Must choose a method in VALID_ENCODINGS")

    if encoding == "sparse":
        train_peptide_encodings = sparse_peptide_encoding(train_peptides)
        train_tap_encodings = sparse_peptide_encoding(
            encode_tap_allele(train_data.Host.values)
        )
    elif encoding == "physico":
        train_peptide_encodings = physico_peptide_encoding(train_peptides)
        train_tap_encodings = physico_peptide_encoding(
            encode_tap_allele(train_data.Host.values)
        )
    else:
        train_peptide_encodings = submatrix_peptide_encoding(
            train_peptides, encoding
        )
        train_tap_encodings = submatrix_peptide_encoding(
            encode_tap_allele(train_data.Host.values), encoding
        )

    train_X = np.hstack([train_peptide_encodings, train_tap_encodings])

    return (
        train_X,
        np.log10(train_data["IC50 (nM)"].values),
        train_data.Length.values,
    )


def encode_tap_peptides(
    peptides: np.ndarray,
    hosts: np.ndarray,
    startpoint: int,
    encoding: str,
    length: int,
):
    fixed_peptides = make_length(peptides, startpoint, length)

    if encoding not in VALID_ENCODINGS:
        raise ValueError("Must choose a method in VALID_ENCODINGS")

    if encoding == "sparse":
        peptide_encodings = sparse_peptide_encoding(fixed_peptides)
        tap_encodings = sparse_peptide_encoding(encode_tap_allele(hosts))
    elif encoding == "physico":
        peptide_encodings = physico_peptide_encoding(fixed_peptides)
        tap_encodings = physico_peptide_encoding(encode_tap_allele(hosts))
    else:
        peptide_encodings = submatrix_peptide_encoding(
            fixed_peptides, encoding
        )
        tap_encodings = submatrix_peptide_encoding(
            encode_tap_allele(hosts), encoding
        )

    X = np.hstack([peptide_encodings, tap_encodings])

    return X


# NETMHCPAN_PSEUDOSEQUENCES = pd.read_csv(
#     "lib/poem/netmhcpan_pseudo.dat", sep="\s+"
# )
# netmhcpan_dict = dict(
#     zip(
#         netmhcpan_pseudosequences["allele"],
#         netmhcpan_pseudosequences["pseudosequence"],
#     )
# )
# g_alphas = pd.read_csv("poem/src/g_alpha.tsv", delimiter="\t")
# g_betas = pd.read_csv("poem/src/g_beta.tsv", delimiter="\t")

# full_sequences = [
#     g_alphas[g_alphas.mhc_allele == a].sequence.values[0].replace("-", "")
#     + g_betas[g_betas.mhc_allele == a].sequence.values[0].replace("-", "")
#     for a in g_alphas.mhc_allele.values[:13350]
# ]
# full_dict = dict(
#     zip(
#         [x.replace("*", "") for x in g_alphas.mhc_allele.values[:13350]],
#         full_sequences,
#     )
# )


# def mhci_allele_to_pseudosequence(allele: str):
#     parsed_allele = mhcgnomes.parse(allele)
#     # sometimes the compact names parse to serotypes, e.g. B3901. I am unsure why.
#     if type(parsed_allele) is mhcgnomes.serotype.Serotype:
#         parsed_allele = parsed_allele.alleles[0].to_string()
#     else:
#         parsed_allele = parsed_allele.to_string()
#     return netmhcpan_dict[parsed_allele.replace("*", "")]


# def allele_to_pseudovec(allele: str, enc: str = "BLOSUM50"):
#     """Returns BLOSUM50 (or other) encoding of allele"""
#     if enc == "sparse":
#         return sparse_peptide_encoding([mhci_allele_to_pseudosequence(allele)])

#     else:
#         return submatrix_peptide_encoding(
#             [mhci_allele_to_pseudosequence(allele)], matrix=enc
#         )


# def allele_to_whole_vec(allele: str, enc: str = "BLOSUM50"):
#     """Returns BLOSUM50 (or other) encoding of allele"""
#     parsed_allele = mhcgnomes.parse(allele)
#     # sometimes the compact names parse to serotypes, e.g. B3901. I am unsure why.
#     if type(parsed_allele) is mhcgnomes.serotype.Serotype:
#         parsed_allele = parsed_allele.alleles[0].to_string()
#     else:
#         parsed_allele = parsed_allele.to_string()
#     parsed_allele = parsed_allele.replace("*", "")
#     if enc == "sparse":
#         return sparse_peptides([full_dict[parsed_allele]])

#     else:
#         return encode_peptides([full_dict[parsed_allele]], matrix=enc)
