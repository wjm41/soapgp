import random
from collections import defaultdict

from tqdm import tqdm
import numpy as np 
from rdkit import Chem
from ase.atoms import Atoms
from itertools import islice
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Set, Tuple, Union


class ConfigASE(object):
    def __init__(self):
        self.info = {}
        self.cell = None
        self.pbc = np.array([False, False, False])
        self.atoms = []
        self.positions = []
        self.symbols = []
    def __len__(self):
        return len(self.atoms)
    def get_positions(self):
        return self.positions
    def get_chemical_symbols(self):
        return self.symbols
    def create(self, n_atoms, fs):
        #header = fs.readline().split()
        # Parse header: key1="str1" key2=123 key3="another value" ...
        header = fs.readline().replace("\n", "")
        tokens = []
        pos0 = 0
        pos1 = 0
        status = "<"
        quotcount = 0
        while pos1 < len(header):
            #print tokens, quotcount, status, pos0, pos1, header[pos0:pos1]
            status_out = status
            # On the lhs of the key-value pair?
            if status == "<":
                if header[pos1] == "=":
                    tokens.append(header[pos0:pos1])
                    pos0 = pos1+1
                    pos1 = pos1+1
                    status_out = ">"
                    quotcount = 0
                else:
                    pos1 += 1
            # On the rhs of the key-value pair?
            elif status == ">":
                if header[pos1-1:pos1] == '"':
                    quotcount += 1
                if quotcount == 0 and header[pos1] == ' ':
                    quotcount = 2
                if quotcount <= 1:
                    pos1 += 1
                elif quotcount == 2:
                    tokens.append(header[pos0:pos1])
                    pos0 = pos1+1
                    pos1 = pos1+1
                    status_out = ""
                    quotcount = 0
                else:
                    assert False
            # In between key-value pairs?
            elif status == "":
                if header[pos1] == ' ':
                    pos0 += 1
                    pos1 += 1
                else:
                    status_out = "<"
            else:
                assert False
            status = status_out
        kvs = []
        for i in range(int(len(tokens)/2)):
            kvs.append([tokens[2*i], tokens[2*i+1]])
        # Process key-value pairs
        for kv in kvs:
            key = kv[0]
            value = '='.join(kv[1:])
            value = value.replace('"','').replace('\'','')
            # Float?
            if '.' in value:
                try:
                    value = float(value)
                except: pass
            else:
                # Int?
                try:
                    value = int(value)
                except: pass
            self.info[kv[0]] = value
        # Read atoms
        self.positions = []
        self.symbols = []
        for i in range(n_atoms):
            ln = fs.readline()
            ln = ln.split()
            name = ln[0]
            pos = list(map(float, ln[1:4]))
            pos = np.array(pos)
            self.positions.append(pos)
            self.symbols.append(name)
        self.positions = np.array(self.positions)
        return

def read_xyz(config_file,
            index=':'):
        species={'C'}
        atom_list = []
        mol_list = []
        num_list = []
        ifs = open(config_file, 'r')
        while True:
            header = ifs.readline().split()
            if header != []:
                assert len(header) == 1
                n_atoms = int(header[0])
                num_list.append(n_atoms)
                config = ConfigASE()
                config.create(n_atoms, ifs)
                atom_list.append(config.get_chemical_symbols())
                atoms = set(config.get_chemical_symbols())
                if (atoms.issubset(species)==False):
                    species = species.union(atoms)
                xyz = config.get_positions()
                mol = Atoms(symbols=config.get_chemical_symbols(), positions= xyz)
                mol_list.append(mol)
            else: break
        return mol_list, num_list, atom_list, species

def parse_dataset(task_name, path, use_fragments=True, subset_size=1000):
    """
    Returns list of molecular smiles, as well as the y-targets of the dataset
    :param task_name: name of the task
    :param path: dataset path
    :param use_fragments: If True return fragments instead of SMILES
    :param subset_size: Subset size for big datasets like CEP or QM9
    :return: x, y where x can be SMILES or fragments and y is the label.
    """

    smiles_list = []
    y = None

    if task_name == 'FreeSolv':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['expt'].to_numpy()  # can change to df['calc'] for calculated values

    elif task_name == 'ESOL':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['measured log solubility in mols per litre'].to_numpy()

    elif task_name == 'lipo':
        df = pd.read_csv(path)
        smiles_list = df['smiles'].tolist()
        y = df['exp'].to_numpy()

    elif task_name == 'CatS':
        df = pd.read_csv(path, usecols=[0, 1], header=None, names=['smiles','y'])
        smiles_list = df['smiles'].tolist()
        y = df['y'].to_numpy()

    elif task_name == 'DLS':
        df = pd.read_csv(path)
        smiles_list = df['SMILES'].tolist()
        y = df['LogS exp (mol/L)'].to_numpy()

    elif task_name == 'bradley':

        df = pd.read_csv(path)
        smiles_list = df['SMILES'].tolist()
        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        good_inds = []
        good_mols = []

        # There are 3025/3042 molecules that can be parsed by RDKit. 3025 is the dataset size commonly reported in the
        # literature cf. the paper:
        # "Bayesian semi-supervised learning for uncertainty-calibrated prediction of molecular properties and
        # active learning"

        for ind, mol in enumerate(rdkit_mols):
            if mol != None:
                good_inds.append(ind)
                good_mols.append(mol)
        df = df.iloc[good_inds]
        smiles_list = df['SMILES'].tolist()
        y = df['Melting Point {measured}'].to_numpy()

    elif task_name == 'Malaria':
        df = pd.read_csv(path, header=0)
        pred_val = 'XC50_3D7 (microM)'
        df = df[((df[pred_val] != 'ND') & (df[pred_val] != '<')) & (df[pred_val].notnull())]
        smiles_list = df['smiles'].tolist()
        y = df[pred_val].to_numpy()
        y = np.log10(y)

    else:
        raise Exception('Must provide valid dataset')
    return smiles_list, y.reshape(-1,1)

def split_by_lengths(seq, num):
    out_list = []
    i=0
    for j in num:
        out_list.append(seq[i:i+j])
        i+=j
    return out_list

def return_borders(index, dat_len, mpi_size):
    mpi_borders = np.linspace(0, dat_len, mpi_size + 1).astype('int')

    border_low = mpi_borders[index]
    border_high = mpi_borders[index+1]
    return border_low, border_high

def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A smiles string or an RDKit molecule.
    :param include_chirality: Whether to include chirality.
    :return:
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)

    return scaffold

def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

def scaffold_split(data: List[str],
                   sizes: Tuple[float, float] = (0.8, 0.2),
                   balanced: bool = False,
                   seed: int = 0):
    """
    Split a dataset by scaffold so that no molecules sharing a scaffold are in the same split.

    :param data: List of smiles strings
    :param sizes: A length-2 tuple with the proportions of data in the
    train  and test sets.
    :param balanced: Try to balance sizes of scaffolds in each set, rather than just putting smallest in test set.
    :param seed: Seed for shuffling when doing balanced splitting.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert sum(sizes) == 1

    # Split
    train_size, test_size = sizes[0] * len(data), sizes[1] * len(data)
    train, test = [], []
    train_scaffold_count, test_scaffold_count = 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(data, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda index_set: len(index_set),
                            reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    #print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
    #                 f'train scaffolds = {train_scaffold_count:,} | '
    #                 f'test scaffolds = {test_scaffold_count:,}')

    # Map from indices to data
    
    #train = [data[i] for i in train]
    #test = [data[i] for i in test]
    #print(train)
    #print(test)
    return train, test

def transform_data(X_train, y_train, X_test, y_test, n_components=None, quiet=False):
    """ 
    Apply feature scaling, dimensionality reduction to the data. Return the standardised and low-dimensional train and
    test sets together with the scaler object for the target values.

    :param X_train: input train data
    :param y_train: train labels
    :param X_test: input test data
    :param y_test: test labels
    :param n_components: number of principal components to keep
    :return: X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
    """

    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)) 
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)) 
    if n_components!=None:
       pca = PCA(n_components)
       X_train_scaled = pca.fit_transform(X_train_scaled)
       if quiet==False:
          print('\nFraction of variance retained is: ' + str(sum(pca.explained_variance_ratio_)))
       X_test_scaled = pca.transform(X_test_scaled)
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler
