# POEM
*POEM project for [@miles-weatherseed](https://github.com/miles-weatherseed) and [@MichaelEBryan](https://github.com/MichaelEBryan) to work on.*

## About

POEM is a Predictor Of (Immunogenic CD8+) Epitopes using Mechanistic modelling of the Class I antigen presentation pathway.

The source code is split into two sections:

1) **Mechanistic Model** - this is a systems biology model of the Class I antigen processing pathway, validated on the experimental datasets of Hearn *et al*, 2010. To calculate peptide sequence specific parameters, various predictive models are used.

2) **POEM** - this is a binary classifier trained using the PRIME2.0 dataset (Gfeller *et al*, 2021) to predict CD8+ immunogenicity. As input features, the predictor uses the ouput of the mechanistic model, along with MHC-I pseudosequence, peptide length, and predicted peptide TCR-interaction properties.


## Quickstart

**NOTE:** *The instructions in this section will not work until the user has followed the installation instructions below.*

To run the mechanistic model, the user needs to have a `.csv` file with the columns:

- `peptide`: containing the peptide sequences to be queried
- `mhc_i`: the name of the binding MHC-I allele
- `fasta`: the filename of the `.fasta` file containing the primary sequence of the corresponding MHC-I allele.

Other columns are permitted but may be overwritten by `length`, `start`, or `predicted_pmhc`. The users should create a directory containing the necessary `.fasta` files.

The mechanistic model can then be simulated by running 

```bash
usage: classi_presentation  [-h] [-p INPUT_PEPTIDES] 
[-d FASTA_DIR] [-c] [-o] [-v]

Required arguments:
  -p INPUT_PEPTIDES, --input_peptides INPUT_PEPTIDES
                        Path to the input peptides CSV file.
  -d FASTA_DIR, --fasta_dir FASTA_DIR
                        Path to the directory containing FASTA files.

Optional arguments:
  -h, --help            show this help message and exit
  -c, --clean_up        Whether or not to clean up the files after completion. Default=False.
  -o {Human,Mouse,Rat_a,Rat_u}, --host_organism {Human,Mouse,Rat_a,Rat_u}
                        Which animal to run the TAP model for. Default=Human.
  -v, --verbose         Whether or not to log progress messages. Default=False.
```

## Installation

POEM is written primarily in Python and should be compatible with Python>=3.7. To avoid conflicts in dependencies, it is recommended that users create a new virtual environment for POEM:

### Using virtual environment

To set up a virtual environment and install the required dependencies from the `requirements.txt` file, follow these steps:

#### 1. Create a Virtual Environment

First, create a virtual environment using `venv`:

```bash
# For Windows:
python -m venv venv

# For macOS/Linux:
python3 -m venv venv
```
#### 2. Activate the Virtual Environment

After creating the virtual environment, activate it

```bash
# For Windows:
.\venv\Scripts\activate

# For macOS/Linux:
source venv/bin/activate
```
#### 3. Install POEM's dependencies

Python dependencies are in the file `requirements.txt`. To install these, run

```bash
pip install -r requirements.txt
```

---

### Installation of POEM using pip

Once the source code has been downloaded and the requisite packages installed, POEM can be installed using pip. From the base directory of POEM, run

```bash
pip install -e .
```

Once installed, POEM's two modules can be run from anywhere on your machine. The mechanistic model can be called using `classi_presentation`. The immunogenicity predictor can be called using `poem`. 

To see the arguments these algorithms take, use the `-h` flag.

---

### Installation of non-proprietry algorithms

POEM also relies upon some external (non-proprietry) algorithms to form its predictions. These must be installed and, where necessary, added to the `PATH`.

#### Installing Pepsickle

Pepsickle is a predictor of proteasomal cleavage (Weeder *et al*, 2021). It can be installed in your virtual environment using:

```bash
pip install pepsickle
```

#### Installing NetMHC and NetMHCpan

NetMHC-4.0 (Andreatta and Nielsen, 2016) and NetMHCpan-4.1 (Reynisson *et al*, 2021) are predictors of peptide-MHC binding affinity. They must be installed from the [DTU website](https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/) by obtaining an appropriate licence for the user (academic or commercial).

Once both packages have been installed, their installation directories should be added to the system's `PATH`. To do so, follow the instructions below:

#### macOS/Linux Users

1. Open your terminal.
2. Run the following command, replacing `/path/to/directory/netMHC-4.0` with the actual installation path:

    ```bash
    # Add NetMHC-4.0 to your PATH temporarily
    export NETMHC_DIR="/path/to/directory/netMHC-4.0"
    export PATH="$NETMHC_DIR:$PATH"
    ```

3. To make this change permanent, add these lines to your shell configuration file:

    **For Zsh users** (default on macOS Catalina and later):

    ```bash
    # Open ~/.zshrc in an editor (e.g., nano)
    nano ~/.zshrc

    # Add the following lines to the end of the file:
    export NETMHC_DIR="/path/to/directory/netMHC-4.0"
    export PATH="$NETMHC_DIR:$PATH"

    # Save and close the file, then apply the changes:
    source ~/.zshrc
    ```

    **For Bash users** (default on most Linux distributions and older versions of macOS):

    ```bash
    # Open ~/.bashrc in an editor (e.g., nano)
    nano ~/.bashrc

    # Add the following lines to the end of the file:
    export NETMHC_DIR="/path/to/directory/netMHC-4.0"
    export PATH="$NETMHC_DIR:$PATH"

    # Save and close the file, then apply the changes:
    source ~/.bashrc
    ```

---

#### Windows Users

1. Press `Win + X` and select **System** or search for **Environment Variables** in the Start Menu and select **Edit the system environment variables**.
2. In the **System Properties** window, click on **Environment Variables**.
3. Under **User variables** or **System variables**, find and select **Path**, then click **Edit**.
4. Click **New** and enter the path to your NetMHC-4.0 installation directory (e.g., `C:\path\to\directory\netMHC-4.0`).
5. Click **OK** to save your changes.

You can now use NetMHC-4.0 from any command prompt window. Repeat this for NetMHCpan-4.1, changing the installation directory as appropriate.