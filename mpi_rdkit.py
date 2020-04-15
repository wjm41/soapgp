import sys
import time

from mpi4py import MPI

import numpy as np
import scipy

from helper import read_xyz, split_by_lengths, return_borders

from rdkit import Chem

from sklearn.preprocessing import normalize

data_name = sys.argv[1]

def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A PyTorch tensor containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

# pd read .can
mol_list = []

mol = Chem.MolFromSmiles(smiles)
atom_list = []
for atom in mol.GetAtoms():
    atom_list.append(atom_features(atom))
mol_list.append(atom_list)
dat_size = len(mols)

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

if mpi_rank==0:
    print("\nEvaluating "+data_name+ " rematch on " + str(mpi_size) + " MPI processes.\n")
    print('No. of molecules = {}\n'.format(dat_size))
    print('Elements present = {}\n'.format(species))


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
    np.save('kernels/'+data_name+'_soap', K) 
    print(K)

mpi_comm.Barrier()
MPI.Finalize()

