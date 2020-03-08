# SOAP-GP for Molecular Property Prediction

This repository contains code for implementing GP regression model with SOAP kernels for QSAR modelling.

## Dependencies

- [dscribe](https://github.com/SINGROUP/dscribe)
- [GPFlow](https://github.com/GPflow/GPflow)
- [RDKit](http://www.rdkit.org/docs/index.html)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- Python packages: numpy, scipy, pandas, scikit-learn


## Workflow

In order to regress molecular properties using SOAP-GP, there are three steps:

1. Data preprocessing using `dataset_prep.sh`
2. Generation of SOAP kernel using `mpi_soap.py`
3. Training of SOAP-GP model using `gpr_soap.py`

## Data
`dataset_prep.py` uses `RDKit` to generate single conformers for each SMILES string in the data file and saves the information to a .sdf file. The `openbabel` package is used to convert the .sdf file to an .xyz file which is used as the input for the calculation of the SOAP kernel.

## SOAP feature calculation


## GP Fitting

## Results
