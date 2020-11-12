import argparse
import random

import numpy as np 
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold

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
    for i, mol in enumerate(mols):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds

def scaffold_split(data: List[str],
                   sizes: Tuple[float, float] = (0.8, 0.2),
                   balanced: bool = True,
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

