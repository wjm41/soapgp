
import numpy as np 
import pandas as pd
from rdkit.Chem import MolFromSmiles


PATHS = {
'FreeSolv': 'data/FreeSolv/FreeSolv.csv', 
'esol': 'data/esol/esol.csv', 
'lipo': 'data/lipo/lipo.csv', 
'dls': 'data/DLS/DLS-100.csv', 
'bradley': 'data/bradley/bradley.csv', 
'IC50': 'data/IC50/'
}
def parse_dataset(task_name, subtask=None):
    """
    Returns list of molecular smiles, as well as the y-targets of the dataset
    :param task_name: name of the task
    :param path: dataset path
    :return: x, y where x is a list of SMILES and y is a numpy array of the target values.
    """

    smiles_list = []
    y = None
 
    path = PATHS[task_name]
 
    if task_name == 'FreeSolv':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['expt'].to_numpy()  # can change to df['calc'] for calculated values

    elif task_name == 'esol':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['measured log solubility in mols per litre'].to_numpy()

    elif task_name == 'dls':
        df = pd.read_csv(path)
        smiles_list = df['SMILES'].tolist()
        y = df['LogS exp (mol/L)'].to_numpy()

    elif task_name == 'lipo':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['exp'].to_numpy()

    elif task_name == 'bradley':
        df = pd.read_csv(path)
        smiles_list = df['SMILES'].tolist()
        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        good_inds = []

        # There are 3025/3042 molecules that can be parsed by RDKit. 3025 is the dataset size commonly reported in the
        # literature cf. the paper:
        # "Bayesian semi-supervised learning for uncertainty-calibrated prediction of molecular properties and
        # active learning"

        for ind, mol in enumerate(rdkit_mols):
            if mol != None:
                good_inds.append(ind)
        df = df.iloc[good_inds]
        smiles_list = df['SMILES'].tolist()
        y = df['Melting Point {measured, converted}'].to_numpy()

    elif task_name == 'IC50':
        path = path + subtask+'.can'
        df = pd.read_csv(path, delim_whitespace=True)
        smiles_list = df['SMILES'].tolist()
        y = df['affinity'].to_numpy()
    else:
        raise Exception('Must provide valid dataset')
    print('length of dataset = '+str(len(y)))
    return smiles_list, y
