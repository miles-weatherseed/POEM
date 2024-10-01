# POEM
POEM project for miles-weatherseed and @MichaelEBryan to work on.

## About

POEM is a Predictor Of (Immunogenic CD8+) Epitopes using Mechanistic modelling of the Class I antigen presentation pathway.

The source code is split into two sections:

1) **Mechanistic Model** - this is a systems biology model of the Class I antigen processing pathway, validated on the experimental datasets of Hearn *et al*, 2010. To calculate peptide sequence specific parameters, various models are used:

    a)  **Proteasomal cleavage product prediction**
    
    Using 

    ii. **Cytosolic aminopeptidase 

## Installation

POEM is written primarily in Python and should be compatible with Python 3.7+. To avoid conflicts in dependencies, it is recommended that users create a new virtual environment for using POEM:

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
