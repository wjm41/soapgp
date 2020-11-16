from collections import defaultdict
from ase.atoms import Atoms
from itertools import islice
import numpy as np

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