import argparse
import numpy as np
import pandas as pd
import mhcgnomes
import os
import sys
import tempfile
import shutil
import yaml
import subprocess
from tqdm import tqdm
from Bio import SeqIO
from mechanistic_model.affinities import get_netmhc, get_netmhcpan
from mechanistic_model.erap1 import get_erap1_kcats
from mechanistic_model.cytosolic_aminopeptidases import get_cytamin_rates
from mechanistic_model.tap import predict_tap_binding_affinities
from mechanistic_model.tapasin_dependence import get_allele_bER
from mechanistic_model.proteasome import (
    get_proteasome_product_probabilities_from_preprocessed,
    run_algos,
)
from mechanistic_model.utils.utils import find_peptide_precursors

current_dir = os.path.dirname(os.path.abspath(__file__))
configuration = yaml.safe_load(
    open(
        os.path.join(current_dir, "data", "mechanistic_model_settings.yml"),
        "r",
    )
)["configuration"]


# Custom type function to enforce case-insensitive comparison
def capitalise_mode(value):
    return value.capitalize()


# Helper function to optionally apply tqdm
def verbose_tqdm(iterator, verbosity, desc=""):
    return tqdm(iterator, desc=desc) if verbosity else iterator


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
        help="Whether or not to clean up the files after completion.",
        action="store_true",
        default=False,
    )

    # Adding -o for host organism
    parser.add_argument(
        "-o",
        "--host_organism",
        help="Which animal to run the TAP model for.",
        default="Human",
        required=False,
        choices=["Human", "Mouse", "Rat_a", "Rat_u"],
        type=capitalise_mode,
    )

    # Adding -v for verbosity
    parser.add_argument(
        "-v",
        "--verbose",
        help="Whether or not to log progress messages.",
        action="store_true",
        default=False,
    )

    # Parse the arguments
    args = parser.parse_args()

    # open input file
    peptides_df = pd.read_csv(args.input_peptides)

    # ensure that the required columns are present
    assert all(
        col in peptides_df.columns for col in ["peptide", "mhc_i", "fasta"]
    ), "DataFrame is missing required columns"

    # add in lengths of peptides if not already present
    peptides_df["length"] = [len(p) for p in peptides_df.peptide]

    # ensure that all MHC-I allele names are consistent with mhcgnomes
    # nomenclature
    peptides_df["mhc_i"] = parse_allele_names(peptides_df.mhc_i.values)

    # load in fasta sequences
    fasta_dictionary = {}
    fasta_names = peptides_df.fasta.unique()
    for f in fasta_names:
        fasta_dictionary[f] = str(
            SeqIO.read(os.path.join(args.fasta_dir, f"{f}.fasta"), "fasta").seq
        )

    # find position of each peptide within its corresponding fasta
    peptides_df["start"] = peptides_df.apply(
        lambda row: fasta_dictionary[row["fasta"]].find(row["peptide"]), axis=1
    )
    if peptides_df["start"].min() < 0:
        raise ValueError(
            "One or more peptide could not be found in the fastas. Check these peptides:",
            peptides_df[peptides_df["start"] < 0].peptide.values,
        )

    # make a temporary directory in which to save newly generated files
    temp_dir = tempfile.mkdtemp(dir=args.fasta_dir)

    # run pepsickle algorithm on FASTAs
    # TODO: decide whether to hold this in memory or in tempfiles
    pepsickle_dictionary = {}
    pepsickle_version = configuration["pepsickle_version"]
    for f in verbose_tqdm(
        fasta_names,
        args.verbosity,
        desc="Running Pepsickle for provided fastas",
    ):
        pepsickle_dictionary[f] = run_algos(
            os.path.join(args.fasta_dir, f"{f}.fasta"),
            pepsickle_version,
            "I",
        )

    # find peptide precursors/products
    peptides_list = []
    peptide_lengths = np.arange(8, configuration["max_length"] + 1)[::-1]
    for i in range(peptides_df.shape[0]):
        peps = find_peptide_precursors(
            peptides_df["length"].values[i],
            fasta_dictionary[peptides_df.fasta.values[i]],
            peptides_df["start"].values[i],
            peptide_lengths,
        )
        peptides_list.append(peps)
    peptides_list = np.array(peptides_list)

    # calculate peptide-specific parameters
    # 1. Proteasomal digestion
    gPs = np.zeros(peptides_list.shape)
    for i in verbose_tqdm(
        range(gPs.shape[0]),
        args.verbosity,
        desc="Calculating proteasomal product probabilities",
    ):
        startpoints = np.array(
            [
                peptides_df["start"].values[i]
                + peptides_df["length"].values[i]
                - x
                for x in peptide_lengths
            ]
        )
        # predict formation probabilities for peptides which physically exist
        predicted_gPs = get_proteasome_product_probabilities_from_preprocessed(
            pepsickle_dictionary[peptides_df["fasta"].values[i]],
            pepsickle_version,
            startpoints[startpoints >= 0],
            peptide_lengths[startpoints >= 0],
        )
        # replace the remaining peptides with zeros
        gPs[i, :] = np.hstack(
            [np.zeros(np.sum([startpoints < 0])), predicted_gPs]
        )
    np.save(os.path.join(temp_dir, "gPs.npy"), gPs)

    # 2. Cytosolic aminopeptidase parameters
    cyt_amin_rates = np.array([get_cytamin_rates(p) for p in peptides_list])
    np.save(os.path.join(temp_dir, "cytamin_out.npy"), cyt_amin_rates)
    np.save(
        os.path.join(temp_dir, "cytamin_in.npy"),
        np.hstack(
            [np.zeros((cyt_amin_rates.shape[0], 1)), cyt_amin_rates[:, :-1]]
        ),
    )

    # 3. TAP binding affinity
    TAP_BAs = np.array(
        [
            predict_tap_binding_affinities(p, args.host_organism)
            for p in verbose_tqdm(
                peptides_list,
                args.verbosity,
                desc="Predicting TAP binding affinities",
            )
        ]
    )
    np.save(os.path.join(temp_dir, "TAP_BAs.npy"), TAP_BAs)

    # 4. ERAP1 kcats
    erap1_kcats = np.array(
        [
            get_erap1_kcats(p)
            for p in verbose_tqdm(
                peptides_list, args.verbosity, desc="Predicting ERAP1 kcats"
            )
        ]
    )
    np.save(os.path.join(temp_dir, "erap1_kcat_out.npy"), erap1_kcats)
    np.save(
        os.path.join(temp_dir, "erap1_kcat_in.npy"),
        np.hstack([np.zeros((erap1_kcats.shape[0], 1)), erap1_kcats[:, :-1]]),
    )

    # 5. MHC-I binding affinity prediction and
    mhci_affinities = np.zeros(peptides_list.shape)
    binding_rates = np.zeros(peptides_list.shape[0])
    affinity_algorithm = configuration["mhci_affinity_algorithm"]
    # batch by MHC-I allele and peptide length
    for allele in verbose_tqdm(
        peptides_df["mhc_i"].unique(),
        args.verbosity,
        desc="Predicting peptide-MHC binding affinities, looping over unique MHC-I alleles",
    ):
        # work out the binding rate using tapasin dependence for this allele
        allele_mask = peptides_df["mhc_i"].values == allele
        binding_rates[allele_mask] = get_allele_bER(allele)
        for col in range(peptides_list.shape[1]):
            if affinity_algorithm == "NetMHC":
                mhci_affinities[allele_mask, col] = get_netmhc(
                    allele,
                    peptides_list[allele_mask, col],
                )
            elif affinity_algorithm == "NetMHCpan":
                mhci_affinities[allele_mask, col] = get_netmhcpan(
                    allele,
                    peptides_list[allele_mask, col],
                )
            else:
                raise ValueError(
                    "Affinity algorithm must currently be either NetMHC or NetMHCpan!"
                )
    np.save(os.path.join(temp_dir, "bERs.npy"), binding_rates)
    np.save(os.path.join(temp_dir, "mhci_affinities.npy"), mhci_affinities)

    # run mechanistic model in Julia
    if args.verbosity is True:
        print("Running mechanistic model in Julia")
    subprocess.run(
        ["julia", os.path.join(current_dir, "mechanistic_model.jl"), temp_dir],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # extract the classi_presentation
    julia_results = np.load(os.path.join(temp_dir, "pmhc_levels.npy"))
    peptides_df["predicted_pmhc"] = julia_results[
        np.arange(len(peptides_df)),
        configuration["max_length"] - peptides_df["length"].values,
    ]

    # save the peptides_df as results
    if args.verbosity is True:
        print("Saving results to poem_mechanistic_results.csv")
    peptides_df.to_csv(
        os.path.join(
            os.path.dirname(args.input_peptides),
            "poem_mechanistic_results.csv",
        ),
        index=False,
    )
    # clean up
    if args.clean_up:
        shutil.rmtree(temp_dir)


def parse_allele_names(allele_names: np.ndarray) -> np.ndarray:
    """Function ensures that allele names conform to mhcgnomes nomenclature

    Args:
        allele_names (np.ndarray): Array of allele names

    Returns:
        np.ndarray: Array of mhcgnomes parsed allele names
    """
    parsed_alleles = []
    for allele_name in allele_names:
        parsed_allele = mhcgnomes.parse(allele_name)
        # sometimes the compact names parse to serotypes, e.g. B3901. I am
        # unsure why.
        if type(parsed_allele) is mhcgnomes.serotype.Serotype:
            parsed_allele = parsed_allele.alleles[0].to_string()
        else:
            parsed_allele = parsed_allele.to_string()
        parsed_alleles.append(parsed_allele)
    return np.array(parsed_alleles)
