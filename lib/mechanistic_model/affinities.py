import numpy as np
from mhctools.netmhc4 import NetMHC4
from mhctools.netmhc_pan41 import NetMHCpan41
import mhctools
import logging

# mhctools has some very annoying logging messages which I have currently
# turned off

logging.getLogger("mhctools").setLevel(logging.CRITICAL)


def get_netmhc(allele_name: str, peptides: np.ndarray) -> np.ndarray:
    """Function returns binding affinities from NetMHC-4.0. Falls to
    NetMHCpan-4.1 if allele is not supported by NetMHC-4.0

    Args:
        allele_name (str): String allele name (e.g. HLA-A*02:01)
        peptides (np.ndarray): Array of peptides to have affinties predicted

    Returns:
        np.ndarray: Array of predicted binding affinities (IC50s in nanomolar)
    """
    # mhctools implementation allows us to parallelise
    try:
        predictor = NetMHC4(alleles=[allele_name], process_limit=-1)
        predictions = predictor.predict_peptides(peptides)
        return np.array([prediction.affinity for prediction in predictions])
    except mhctools.unsupported_allele.UnsupportedAllele:
        # fall back to NetMHCpan
        return get_netmhcpan(allele_name, peptides)


def get_netmhcpan(allele_name: str, peptides: np.ndarray) -> np.ndarray:
    """Function returns binding affinities from NetMHCpan-4.1

    Args:
        allele_name (str): String allele name (e.g. HLA-A*02:01)
        peptides (np.ndarray): Array of peptides to have affinties predicted

    Returns:
        np.ndarray: Array of predicted binding affinities (IC50s in nanomolar)
    """
    # mhctools implementation allows us to parallelise
    predictor = NetMHCpan41(alleles=[allele_name], process_limit=-1)
    predictions = predictor.predict_peptides(peptides)

    return np.array([prediction.affinity for prediction in predictions])


if __name__ == "__main__":
    print(get_netmhc("HLA-A*03:10", np.array(["SIINFEKLM"])))
