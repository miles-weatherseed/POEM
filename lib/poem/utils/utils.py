import numpy as np
import pandas as pd
import mhcgnomes
import os
from Bio.Align import substitution_matrices

current_dir = os.path.dirname(os.path.abspath(__file__))

SUBS_MATRICES = substitution_matrices.load()
VALID_ENCODINGS = [x.lower() for x in SUBS_MATRICES] + [
    "sparse",
]

# load in log enrichment scores from NetTepi (Calis et el.)
enrichment_data = pd.read_csv(
    os.path.join(current_dir, "data", "NetTepi_Enrichments.csv")
)
AA_scores = dict(
    zip(enrichment_data.AA.values, enrichment_data.log_enrichment_score.values)
)
AA_scores["X"] = 0

# define AA order for PRIME2.0 encoding
AAs = [
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
]
aa_order = dict(zip(AAs, np.arange(20)))


def encode_nettepi_tcr_residues(peplist: np.ndarray) -> np.ndarray:
    """Encodes P4 --> P9 using method of NetTepi (Calis log enrichment sores)

    Args:
        peplist (np.ndarray): Array of peptides to be encoded

    Returns:
        np.ndarray: Array of encoded peptides, size (len(peplist), 6)
    """
    pred = np.zeros((len(peplist), 6))
    for n, row in enumerate(peplist):
        peptide = row[0]
        immuscore = np.zeros(6)
        if len(peptide) == 9:
            for i in range(3, 9):
                immuscore[i - 3] = AA_scores[peptide[i - 1]]
            pred[n, :] = immuscore
        else:
            list_9mers = convert_to_9mers(peptide)
            score_sums = np.zeros(6)
            for p in list_9mers:
                p_scores = np.zeros(6)
                for i in range(3, 9):
                    p_scores[i - 3] = AA_scores[p[i - 1]]
                score_sums += p_scores
            immuscore = score_sums / len(list_9mers)
            pred[n, :] = immuscore
    return pred


def convert_to_9mers(peptide: str) -> list[str]:
    """_summary_

    Args:
        peptide (str): _description_

    Returns:
        list[str]: _description_
    """
    list_9mers = []
    length = len(peptide)
    if length < 9:
        insert = "X" * (9 - length)
        for i in range(3, length):
            p = peptide[:i] + insert + peptide[i:]
            list_9mers.append(p)
    elif length == 9:
        list_9mers.append(peptide)
    else:
        for i in range(3, 9):
            p = peptide[:i] + peptide[i + length - 9 :]
            list_9mers.append(p)
    return list_9mers


def get_mia_positions(allele: str, length: int) -> tuple[list, int]:
    """Returns the MIA positions for the restricting MHC-I allele and peptide
    length. Adapted from PRIME2.0 source code.

    Args:
        allele (str): MHC-I allele short name (e.g. A0201)
        length (int): Peptide length

    Returns:
        tuple[list, int]: Indices and number of MIA residues
    """
    # returns list of postitions and a count of the number of positions
    pos = []
    ct = 0

    if allele in [
        "B0801",
        "B1401",
        "B1402",
        "B3701",
        "A6802",
        "B4701",
        "H-2-Db",
    ]:
        pos.append(3)
        ct += 1
        for j in range(5, length - 1):
            pos.append(j)
            ct += 1
    elif allele in ["A0201", "A0202", "A0207", "A0211"]:
        for j in range(4, length - 1):
            pos.append(j)
            ct += 1
    elif allele in ["A0203", "A0204", "A0205", "A0206", "A0220"]:
        for j in range(4, length - 4):
            pos.append(j)
            ct += 1
        pos.append(length - 3)
        ct += 1
        pos.append(length - 2)
        ct += 1
    elif allele in ["A2501", "A2601"]:
        for j in range(4, length - 4):
            pos.append(j)
            ct += 1
        pos.append(length - 3)
        ct += 1
    elif allele in [
        "A2902",
        "B1517",
        "B5802",
        "C0602",
        "C0704",
        "C1502",
        "C1505",
        "C1602",
        "C1701",
        "G0101",
        "G0103",
        "G0104",
    ]:
        for j in range(4, length - 3):
            pos.append(j)
            ct += 1
        pos.append(length - 2)
        ct += 1
    elif allele in [
        "A3201",
        "B1803",
        "B3901",
        "B3905",
        "B3906",
        "B3924",
        "B4601",
        "B4801",
        "C0102",
    ]:
        for j in range(3, length - 2):
            pos.append(j)
            ct += 1
    elif allele == "H-2-Kb":
        pos.append(3)
        ct += 1
        for j in range(6, length - 1):
            pos.append(j)
            ct += 1
    else:
        for j in range(3, length - 1):
            pos.append(j)
            ct += 1

    return pos, ct


def encode_prime2_tcr_residues(
    peptide: str, allele: str, plen: int
) -> np.ndarray:
    """Returns encoding of peptide's TCR-interaction residues using PRIME2.0
    method.

    Args:
        peptide (str): Peptide sequence
        allele (str): Restricting MHC-I allele name (in any format)
        plen (int): Length of peptide

    Returns:
        np.ndarray: Array of encoding.
    """
    # convert allele name to A0101 format
    allele = mhcgnomes.parse(allele).compact_string()
    positions, npos = get_mia_positions(allele, plen)
    tcr_enc = np.zeros(20)
    for p in positions:
        tcr_enc[aa_order[peptide[p]]] += 1
    return tcr_enc / npos


def make_poem_input_layer(
    alleles: np.ndarray,
    plens: np.ndarray,
    peptides: np.ndarray,
    mech_output: np.ndarray,
    encode_length: bool = True,
    mhci_encoding: str = "blosum50",
    mhci_rep: str = "pseudo",
    tcr_encoding: str = "nettepi",
) -> np.ndarray:
    """Produces input features for POEM.

    Args:
        alleles (np.ndarray): Array of MHC-I alleles
        plens (np.ndarray): Array of peptide lengths
        peptides (np.ndarray): Array of peptides
        mech_output (np.ndarray): Array of pMHC levels from mechanistic model
        encode_length (bool): Whether or not to include a sparse peptide length
        encoding
        mhci_encoding (str, optional): The method by which to encode the MHC-I
        allele. Defaults to "BLOSUM50".
        mhci_rep (str, optional): The type of sequence to take for MHC-I.
        Defaults to "pseudo".
        tcr_encoding (str, optional): The method by which to encode the TCR
        interaction residues. Defaults to "NetTepi".

    Returns:
        np.ndarray: Returns the input features for POEM
    """

    # Encode MHC-I alleles
    if mhci_rep == "pseudosequence":
        allele_vecs = np.vstack(
            [
                mhci_allele_to_pseudovec(allele, enc=mhci_encoding)
                for allele in alleles
            ]
        )
    elif mhci_rep == "full":
        allele_vecs = np.vstack(
            [
                mhci_allele_to_whole_vec(allele, enc=mhci_encoding)
                for allele in alleles
            ]
        )
    elif mhci_rep == "none":
        allele_vecs = np.zeros((len(alleles), 0))

    # Encode TCR interaction residues
    if tcr_encoding == "nettepi":
        tcr_encoded_peptides = encode_nettepi_tcr_residues(
            peptides.reshape(-1, 1)
        )
    elif tcr_encoding == "prime":
        tcr_encoded_peptides = np.stack(
            [
                encode_prime2_tcr_residues(peptides[i], alleles[i], plens[i])
                for i in range(len(peptides))
            ]
        )
    elif tcr_encoding == "none":
        # create a width 0 array
        tcr_encoded_peptides = np.zeros((len(peptides), 0))

    # Create length encoding
    if encode_length:
        length_encodings = np.zeros((len(alleles), 9))
        length_encodings[np.arange(len(alleles)), 16 - plens] = 1
    else:
        length_encodings = np.zeros((len(alleles), 0))

    # Compute log10 of mechanistic output
    log_mech_output = np.log(mech_output)

    # Concatenate all parts to form the input layer
    input_layer = np.hstack(
        [
            allele_vecs,
            length_encodings,
            tcr_encoded_peptides,
            log_mech_output[:, np.newaxis],
        ]
    )

    return input_layer
