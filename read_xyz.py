from collections import defaultdict
from ase.atoms import Atoms
from itertools import islice

from typing import Dict, List, Set, Tuple, Union
def read_xyz(config_file,
            index=':'):
        species=set()
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

def str2bool(v):
    """
    Parameters
    ----------
    v
    Returns
    -------
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class ConfigASE(object):
    def __init__(self):
        self.info = {}
        self.cell = None
        self.pbc = np.array([False, False, False])
        self.atoms = []
        self.positions = []
        self.symbols = []
