import sys
import time

from mpi4py import MPI

import numpy as np
import scipy

from helper import read_xyz, split_by_lengths, return_borders

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel

from sklearn.preprocessing import normalize

data_name = sys.argv[1]

mols, num_list, atom_list, species = read_xyz('data/'+data_name+"/"+data_name+".xyz")

dat_size = len(mols)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

if mpi_rank==0:
    print("\nEvaluating "+data_name+ " rematch on " + str(mpi_size) + " MPI processes.\n")
    print('No. of molecules = {}\n'.format(dat_size))
    print('Elements present = {}\n'.format(species))

# Setting up the SOAP descriptor
rcut_small = 3.0
sigma_small = 0.2
rcut_large = 6.0
sigma_large = 0.4

small_soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut_small,
    nmax=12,
    lmax=8,
    sigma = sigma_small,
    sparse=True
)

large_soap = SOAP(
    species=species,
    periodic=False,
    rcut=rcut_large,
    nmax=12,
    lmax=8,
    sigma = sigma_large,
    sparse=True
)

t0 = time.time()
my_border_low, my_border_high = return_borders(mpi_rank, dat_size, mpi_size)

my_mols = mols[my_border_low:my_border_high]
soap = scipy.sparse.hstack([small_soap.create(my_mols),large_soap.create(my_mols)])

t1 = time.time()
if mpi_rank==0:
   print("SOAP: {:.2f}s\n".format(t1-t0))
   print("rcut_small = {:.1f}, sigma_small = {:.1f}, rcut_large = {:.1f}, sigma_large = {:.1f}".format(rcut_small,sigma_small,rcut_large,sigma_large))

soap = normalize(soap, copy=False)

my_soap =  split_by_lengths(soap, num_list[my_border_low:my_border_high])

my_len = len(my_soap)

t2 = time.time()
if mpi_rank==0:
   print ("Normalise & Split Descriptors: {:.2f}s\n".format(t2-t1))

re = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.5, threshold=1e-6, normalize_kernel=True)

K = np.zeros((my_len, dat_size), dtype=np.float32)
sendcounts = np.array(mpi_comm.gather(my_len*dat_size,root=0))

if mpi_rank==0:
   K_full = np.empty((dat_size,dat_size),dtype=np.float32)
   print("K memory usage(bytes): {}".format(K.nbytes+K_full.nbytes))
else:
   K_full = None

#row-parallelised kernel computation
for index in range(0, mpi_size):
    if index==mpi_rank: 
       K[:, my_border_low:my_border_high] += re.create(my_soap).astype(np.float32)
       continue #skip useless calculation

    start, end = return_borders(index, dat_size, mpi_size)
    ref_mols = mols[start:end]
    ref_soap = scipy.sparse.hstack([small_soap.create(ref_mols),large_soap.create(ref_mols)])
    ref_soap = normalize(ref_soap, copy=False)
    ref_soap = split_by_lengths(ref_soap, num_list[start:end])
    K[:, start:end] += re.create(my_soap, ref_soap).astype(np.float32)

#Gather kernel rows
mpi_comm.Gatherv(sendbuf=K,recvbuf = (K_full, sendcounts),root=0)

K = K_full

if mpi_rank==0:
    t3 = time.time()
    print ("Normalised Kernel: {:.2f}s\n".format(t3-t2))

    #Change directory name if you have memory constraints
    np.save('kernels/'+data_name+'_kernel', K) 
    print(K)

mpi_comm.Barrier()
MPI.Finalize()

