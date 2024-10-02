import argparse
from mechanistic_model.affinities import get_netmhc, get_netmhcpan
from mechanistic_model.erap1 import get_erap1_kcats
from mechanistic_model.cytosolic_aminopeptidases import get_cytamin_rates
from mechanistic_model.tap import predict_tap_binding_affinities
from mechanistic_model.tapasin_dependence import get_allele_bER
from mechanistic_model.proteasome import get_proteasome_product_probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Process input_peptides and fasta_dir."
    )

    # Adding -p for input_peptides (CSV file)
    parser.add_argument(
        "-p",
        "--input_peptides",
        help="Path to the input peptides CSV file.",
        required=True,
    )

    # Adding -d for fasta_dir (directory containing FASTA files)
    parser.add_argument(
        "-d",
        "--fasta_dir",
        help="Path to the directory containing FASTA files.",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Logic for handling the arguments
    print(f"Input Peptides: {args.input_peptides}")
    print(f"FASTA Directory: {args.fasta_dir}")
