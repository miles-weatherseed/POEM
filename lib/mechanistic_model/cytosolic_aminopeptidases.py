import numpy as np

# Maximum A Posteriori parameters from fitting data in M_Weatherseed PhD thesis
cyt_amin_dict = {
    "R": 0.06638477867795106,
    "N": 0.012081648881765047,
    "D": 0.0006308219647091677,
    "E": 0.001668333188416447,
    "G": 0.0006121424762350458,
    "H": 0.0077035064796737406,
    "L": 0.059440148429491635,
    "K": 0.0488351773072918,
    "M": 0.03817939903696182,
    "S": 0.02237175963513948,
    "T": 0.01611333010965043,
    "W": 0.23108002348193513,
    "Y": 0.023854165910361415,
    "V": 0.024533763938376123,
    "A": 0.05361892463897976,
    "C": 0.017075435988791222,
    "I": 0.008777466675203379,
    "F": 0.01729911761080652,
    "Q": 0.013233409227650567,
    "P": 0.006545404187764979,
    "X": 0.03350193789235774,
}


def get_cytamin_rates(peptides: np.ndarray) -> np.ndarray:
    """Returns the trimming rate by cytosolic aminopeptidases based on the
    N-terminus of the peptides

    Args:
        peptides (np.ndarray): Array containing peptides to predict cyt_amin
        rates for

    Returns:
        np.ndarray: Trimming rates (s^-1)
    """
    return np.array([cyt_amin_dict[p[0]] for p in peptides])
