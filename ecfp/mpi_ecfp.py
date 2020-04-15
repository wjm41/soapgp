import sys
import numpy as np

from mpi4py import MPI
from rdkit.Chem import MolFromSmiles, AllChem
from e3fp.fingerprint.metrics.array_metrics import tanimoto

from helper import split_by_lengths, return_borders, parse_dataset

FREESOLV_PATH = '../data/FreeSolv/FreeSolv.csv'
CATS_PATH = '../data/CatS/CatS.csv'
LIPO_PATH = '../data/lipo/lipo.csv'
ESOL_PATH = '../data/esol/esol.csv'
DLS_PATH = '../data/DLS/DLS-100.csv'
BRADLEY_PATH = '../data/bradley/bradley.csv'
MALARIA_PATH = '../data/Malaria/Malaria.csv'
CHEMBL5118_PATH = '../data/CHEMBL5118.csv'
CHEMBL3927_PATH = '../data/CHEMBL3927.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'DLS': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH, 'CHEMBL5118': CHEMBL5118_PATH, 'CHEMBL3927' : CHEMBL3927_PATH}

task = sys.argv[1]
#bits = int(sys.argv[2])
#TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'dls', 'CEP', 'CatS', 'bradley', 'Malaria']

smiles_list, y  = parse_dataset(task, PATHS[task]) #NEED TO FIX MALARIA

dat_size = len(smiles_list)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

my_border_low, my_border_high = return_borders(mpi_rank, dat_size, mpi_size)

my_list = smiles_list[my_border_low:my_border_high]

bit_list = [2048]
for bits in bit_list:        
    my_mols = [MolFromSmiles(smiles) for smiles in my_list]
    X = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=bits) for mol in my_mols]
    X = np.asarray(X)
    
    dbs = mpi_comm.gather(X, root=0) 
    
    if mpi_rank==0:
       for db in dbs[1:]:
           X = np.vstack([X, db])
       #print(X)
       #np.save('data/'+task+'/'+task+'_ecfp_'+str(bits)+'.npy',X)
       
       print('Number of bits: {}'.format(bits))
       kernel = 1-tanimoto(X,X)
       print(kernel)
       print(kernel.shape)
       np.save('../kernels/'+task+'_ecfp_'+str(bits)+'.npy',kernel)
    
    mpi_comm.Barrier()
MPI.Finalize()
