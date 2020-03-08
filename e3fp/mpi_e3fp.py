import sys
import numpy as np

from mpi4py import MPI
from e3fp.fingerprint.db import FingerprintDatabase, append
from e3fp.fingerprint.metrics import tanimoto

from helper import split_by_lengths, return_borders, parse_dataset
from utils_e3fp import gen_e3fp_features

# Adjust accordingly for your own file system
FREESOLV_PATH = 'data/FreeSolv/FreeSolv.csv'
CATS_PATH = 'data/CatS/CatS.csv'
LIPO_PATH = 'data/lipo/lipo.csv'
ESOL_PATH = 'data/esol/esol.csv'
DLS_PATH = 'data/dls/dls.csv'
BRADLEY_PATH = 'data/bradley/bradley.csv'
MALARIA_PATH = 'data/Malaria/Malaria.csv'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'dls': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH}

task = sys.argv[1]
#TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'dls', 'CEP', 'CatS', 'bradley', 'Malaria']

smiles_list, y  = parse_dataset(task, PATHS[task]) #NEED TO FIX MALARIA

dat_size = len(smiles_list)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

my_border_low, my_border_high = return_borders(mpi_rank, dat_size, mpi_size)

my_list = smiles_list[my_border_low:my_border_high]

bit_list = [512,1024,2048,4096,8192]
for bits in bit_list:        
  my_db = gen_e3fp_features(my_list, mpi_rank,mpi_size, bits)

  dbs = mpi_comm.gather(my_db, root=0)

  db_list = []
  if mpi_rank==0:
     for db in dbs:
         db_list.append(db)
  
     database = append(db_list)
     #print(database)
     #database.save('data/'+task+'/'+task)
     
     #db = FingerprintDatabase.load('data/'+task+'/'+task+'.fps.bz2')
     #print(db)
     
     kernel = 1-tanimoto(database)
     print(kernel)
     print(kernel.shape)
     #np.save('kernels/'+task+'_e3fp.npy',kernel)
     np.save('/rds-d2/user/wjm41/hpc-work/kernel/e3fp/'+task+'_'+str(bits)+'.npy',kernel)
  mpi_comm.Barrier()
MPI.Finalize()
