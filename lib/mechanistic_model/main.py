import argparse
import numpy as np
import pandas as pd
from mechanistic_model.affinities import get_netmhc, get_netmhcpan
from mechanistic_model.erap1 import get_erap1_kcats
from mechanistic_model.cytosolic_aminopeptidases import get_cytamin_rates
from mechanistic_model.tap import predict_tap_binding_affinities
from mechanistic_model.tapasin_dependence import get_allele_bER
from mechanistic_model.proteasome import get_proteasome_product_probabilities
from mechanistic_model.utils.utils import find_peptide_precursors


def main():
    parser = argparse.ArgumentParser(
        description="Process input_peptides and fasta_dir."
    )

    # Adding -p for input_peptides (CSV file)
    parser.add_argument(
        "-p",
        "--input_peptides",
        help="Path to the input peptides CSV file.",
        type=str,
        required=True,
    )

    # Adding -d for fasta_dir (directory containing FASTA files)
    parser.add_argument(
        "-d",
        "--fasta_dir",
        help="Path to the directory containing FASTA files.",
        type=str,
        required=True,
    )

    # Adding -c for clean_up
    parser.add_argument(
        "-c",
        "--clean_up",
        help="Whether or not to clean up the files after completion (default=True)",
        type=bool,
        default=True,
        required=False,
    )

    # Parse the arguments
    args = parser.parse_args()

    # open input file and add in lengths of peptides if not already present
    peptides_df = pd.read_csv(args.input_peptides)
    peptides_df["length"] = [len(p) for p in peptides_df.peptide]

    # load in fasta sequences to find positions of peptides

    # make a temporary directory in which to save newly generated files

    # run pepsickle algorithm on FASTAs

    # calculate peptide-specific parameters

    # run mechanistic model in Julia

    # clean up


# def process_peptides_df(peptides_df: pd.DataFrame) -> pd.DataFrame:
#     """Add in start and end positions, along with peptide lengths to the dataframe

#     Args:
#         peptides_df (pd.DataFrame): The loaded data frame from the main() function

#     Returns:
#         pd.DataFrame: The updated data frame
#     """

# def calculate_peptide_parameters(peptides_df: pd.DataFrame) ->
