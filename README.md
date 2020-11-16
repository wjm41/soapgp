# SOAP-GP for Molecular Property Prediction

This repository contains code for implementing GP regression models with SOAP kernels for QSAR modelling, as detailed in ["Investigating 3D Atomic Environments for Enhanced QSAR", W. McCorkindale, C. Poelking, A. A. Lee, 2020](https://arxiv.org/abs/2010.12857).

## Dependencies

- [dscribe](https://github.com/SINGROUP/dscribe)
- [GPFlow](https://github.com/GPflow/GPflow)
- [RDKit](http://www.rdkit.org/docs/index.html)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/)
- Python packages: numpy, scipy, pandas, scikit-learn


## Workflow

In order to regress molecular properties using SOAP-GP, there are three steps:

1. Data preprocessing using `dataset_prep.sh`
2. Generation of SOAP descriptors/kernel using `mpi_soap.py`
3. Training of SOAP-GP model using `gpr_soap.py` or `gpr_onthefly.py`

## Data
`dataset_prep.py` uses `RDKit` to generate single conformers for each SMILES string in the data file and saves the information to a .sdf file. The `openbabel` package is used to convert the .sdf file to an .xyz file which is used as the input for the calculation of the SOAP kernel.

## SOAP feature generation
The `dscribe` package is used to calculate the SOAP vectors from the .xyz files generated above using the script `mpi_soap.py`. 

There are two options: you can save the SOAP vectors directly by calling the `-save_soap` argument (for use with `gpr_onthefly.py`) or you can save the SOAP kernel (on by default with `-save_kernel=True` for use with `gpr_soap.py`). 

Calculation of the SOAP kernel is implemented with MPI parallelization as it is quite time-consuming (for a dataset of several thousand molecules ~15 minutes parallelized over 32 cores).

The kernel calculation can also be very very memory consuming - if you get an error signal from MPI indicating `... exited on signal 9 (Killed)` that means one of your MPI processes has ran out of memory! Either throw the job on a cluster with more cores, or contemplate running `gpr_onthefly.py` instead.
 
## GP Fitting
Two scripts are included for demonstrating how to train SOAP-GP for learning molecular properties. 

For benchmarking on a known dataset over multiple splits, typically with a large training set and a small held-out test set, it is more efficient to generate the SOAP kernel in one go (via `mpi_soap.py`) and train GP models that use slice rows/columns from that kernel - that is implemented in `gpr_soap.py`.

In a real-world drug discovery setting where one is only able to train on a tiny dataset but has to screen a large library, generating the full NxN kernel matrix is wasteful and would probably lead to memory issues. In this case it would be better to calculate the relevant kernel entries on-the-fly which is implemented in `gpr_onthefly.py`. The downside is that training/inference will likely take much much longer this way.

As of Nov 2020 the GPFlow website seems to be ambiguous on the feasibility of saving GPFlow models without converting to a `tf.function` (preventing eager execution which thus breaks the SOAP kernel functions) - a manual workaround I have found is to just tell the model to print the kernel parameters with `print_summary` (turned on by `log=True` within the code) and initialize a new kernel with those parameters.
