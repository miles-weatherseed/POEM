import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
import os
import mhcgnomes

# Get the absolute path to the directory where this script resides
current_dir = os.path.dirname(os.path.abspath(__file__))


def encode_to_blosum62(aa_string: str) -> np.ndarray:
    """Snippet to convert a string of amino acids to BLOSUM62 encoding

    Args:
        aa_string (str): string of amino acids

    Returns:
        np.ndarray: Encoding of the input string using BLOSUM62
    """
    blosum62 = substitution_matrices.load("BLOSUM62")
    return np.hstack([blosum62[aa].values() for aa in aa_string])


def TD_to_bER(TD: float) -> float:
    """Takes in TD and returns log10 of the binding rate required to achieve
    this in the mechanistic model.

    Args:
        TD (float): Tapasin dependence

    Returns:
        float: log10 of the binding rate, bER (N.B. usually around 1e-9)
    """
    bERs = np.load("lib/mechanistic_model/data/bERs.npy")
    TDs = np.load("lib/mechanistic_model/data/TDs.npy")
    return np.interp(TD, np.flip(TDs), np.flip(bERs))


def get_nearest_neighbour_TD(
    test_mhci_blosum62: np.ndarray,
    bashirova_hlas_blosum62: np.ndarray,
    bashirova_TDs: np.ndarray,
) -> float:
    """Function takes in a BLOSUM62 encoded MHC-I, finds nearest neighbour in
    Bashirova et al dataset by minimum BLOSUM62 distance, and returns the
    tapasin dependence of the nearest neighbour.

    Args:
        test_mhci_blosum62 (np.ndarray): BLOSUM62 encoding of test MHC-I sequence
        bashirova_hlas_blosum62 (np.ndarray): BLOSUM62 encoding of the alleles in the Bashirova dataset
        bashirova_TDs (np.ndarray): array of the tapasin dependencies in the Bashirova dataset

    Returns:
        float: Tapasin dependence
    """
    # find index of the nearest neighbour in L2 norm (or L1??)
    differences = bashirova_hlas_blosum62 - test_mhci_blosum62
    squared_differences = differences**2
    # Sum the squared differences along the columns to get the squared L2 distance
    squared_distances = np.sum(squared_differences, axis=1)
    # Find the index of the row with the minimum squared distance
    min_index = np.argmin(squared_distances)
    # return the corresponding TD
    return bashirova_TDs[min_index]


def create_TD_database():
    # numpy files with bERs and corresponding TDs from simulating mechanistic model
    MHC_seq_database = pd.read_csv(
        os.path.join(current_dir, "data", "netmhcpan_pseudo.dat"), sep=r"\s+"
    )
    bashirova_data = pd.read_csv(
        os.path.join(current_dir, "data", "bashirova_database.csv")
    )
    bashirova_alleles = []
    for a in bashirova_data.Allele.values:
        parsed_allele = mhcgnomes.parse(a)
        # sometimes the compact names parse to serotypes, e.g. B3901. I am unsure why.
        if type(parsed_allele) is mhcgnomes.serotype.Serotype:
            parsed_allele = parsed_allele.alleles[0].to_string()
        else:
            parsed_allele = parsed_allele.to_string()
        bashirova_alleles.append(parsed_allele)
    bashirova_TDs = bashirova_data.TD.values
    bashirova_loci = np.array([a[0] for a in bashirova_data.Allele.values])
    bashirova_hlas_blosum62 = np.vstack(
        [
            encode_to_blosum62(
                MHC_seq_database[
                    MHC_seq_database.allele == a
                ].pseudosequence.values[0]
            )
            for a in bashirova_alleles
        ]
    )
    TD_list = []
    bER_list = []
    for a, seq in zip(
        MHC_seq_database.allele.values, MHC_seq_database.pseudosequence.values
    ):
        test_blosum62 = encode_to_blosum62(seq)
        if a[:5] in [f"HLA-{locus}" for locus in ["A", "B", "C"]]:
            # I think we should only compare like for like for HLA
            nn_TD = get_nearest_neighbour_TD(
                test_blosum62,
                bashirova_hlas_blosum62[bashirova_loci == a[4], :],
                bashirova_TDs[bashirova_loci == a[4]],
            )
        else:
            nn_TD = get_nearest_neighbour_TD(
                test_blosum62,
                bashirova_hlas_blosum62,
                bashirova_TDs,
            )
        nn_bER = TD_to_bER(nn_TD)
        TD_list.append(nn_TD)
        bER_list.append(nn_bER)
    MHC_seq_database["TD"] = TD_list
    MHC_seq_database["bER"] = bER_list
    MHC_seq_database.to_csv(
        os.path.join(
            current_dir, "cache", "tapasin_dependence", "allele_to_bER.csv"
        ),
        index=False,
    )
    return


# upon installation, create the larger database using nearest neighbour
if not os.path.exists(
    os.path.join(
        current_dir, "cache", "tapasin_dependence", "allele_to_bER.csv"
    ),
):
    create_TD_database()
TD_database = pd.read_csv(
    os.path.join(
        current_dir, "cache", "tapasin_dependence", "allele_to_bER.csv"
    ),
)


def get_allele_bER(allele_name: str) -> float:
    """Uses `mhcgnomes` to standardise the allele name, then returns the log10
    bER of the nearest neighbour by BLOSUM62 distance between the pseudoseqs.

    Args:
        allele_name (str): Allele name in any format (e.g. A0201 or HLA-A*02:01)

    Returns:
        float: The nearest neighbour log10 peptide-MHC binding rate.
    """
    parsed_allele = mhcgnomes.parse(allele_name)
    # sometimes the compact names parse to serotypes, e.g. B3901. I am unsure why.
    if type(parsed_allele) is mhcgnomes.serotype.Serotype:
        parsed_allele = parsed_allele.alleles[0].to_string()
    else:
        parsed_allele = parsed_allele.to_string()
    return TD_database.loc[TD_database.allele == parsed_allele, "bER"].item()


if __name__ == "__main__":
    print(get_allele_bER("HLA-A*02:01"))
    print(get_allele_bER("HLA-A02:01"))
    print(get_allele_bER("A0201"))
    print(get_allele_bER("A6901"))
