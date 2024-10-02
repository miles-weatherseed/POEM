import argparse
from affinities import get_netmhc, get_netmhcpan
from erap1 import get_kcats
from cytosolic_aminopeptidases import get_cytamin_rates
from tap import predict_tap_binding_affinities
from tapasin_dependence import get_allele_bER
from proteasome import get_proteasome_product_probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Process input_peptides and fasta_dir."
    )
    parser.add_argument(
        "input_peptides", help="Path to the input peptides CSV file."
    )
    parser.add_argument(
        "fasta_dir", help="Path to the directory containing FASTA files."
    )

    args = parser.parse_args()

    input_peptides = args.input_peptides
    fasta_dir = args.fasta_dir

    print(f"Input peptides CSV path: {input_peptides}")
    print(f"FASTA directory path: {fasta_dir}")


if __name__ == "__main__":
    main()
