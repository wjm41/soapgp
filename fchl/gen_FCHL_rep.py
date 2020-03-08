import argparse
import os
import sys

import numpy as np
from ase.io import read, write
from sklearn.decomposition import PCA
from qml.representations import generate_fchl_acsf
#from qml.fchl import generate_representation, get_local_kernels
from qml.kernels import get_local_kernels, get_global_kernel

from helper import str2bool

# Adjust accordingly for your own file system
FREESOLV_PATH = 'data/FreeSolv/FreeSolv.xyz'
CATS_PATH = 'data/CatS/CatS.xyz'
LIPO_PATH = 'data/lipo/lipo.xyz'
ESOL_PATH = 'data/esol/esol.xyz'
DLS_PATH = 'data/dls/dls.xyz'
BRADLEY_PATH = 'data/bradley/bradley.xyz'
MALARIA_PATH = 'data/Malaria/Malaria.xyz'

PATHS = {'FreeSolv': FREESOLV_PATH, 'esol': ESOL_PATH, 'lipo': LIPO_PATH, 'dls': DLS_PATH, 'CatS':CATS_PATH, 'bradley':BRADLEY_PATH, 'Malaria':MALARIA_PATH}

TASK_NAME = 'FreeSolv'  # Change dataset. Options: ['ESOL', 'FreeSolv', 'QM9', 'CEP', 'CatS', 'Melt', 'Malaria']





def repr_wrapper(frame, elements, is_periodic = False,
                                        nRs2=24, nRs3=20,
                                        nFourier=1, eta2=0.32, eta3=2.7,
                                        zeta=np.pi, rcut=8.0, acut=8.0,
                                        two_body_decay=1.8, three_body_decay=0.57,
                                        three_body_weight=13.4, pad=False):
    '''
   Periodic systems not implemented for FCHL19.
    :frame: ase Atoms class
    :param elements: list of unique nuclear charges (atom types)
    :type elements: numpy array
    :is_periodic: Boolean determining Whether the system is periodic.
    :type Boolean:
    :param nRs2: Number of gaussian basis functions in the two-body terms
    :type nRs2: integer
    :param nRs3: Number of gaussian basis functions in the three-body radial part
    :type nRs3: integer
    :param nFourier: Order of Fourier expansion
    :type nFourier: integer
    :param eta2: Precision in the gaussian basis functions in the two-body terms
    :type eta2: float
    :param eta3: Precision in the gaussian basis functions in the three-body radial part
    :type eta3: float
    :param zeta: Precision parameter of basis functions in the three-body angular part
    :type zeta: float
    :param two_body_decay: exponential decay for the two body function
    :type two_body_decay: float
    :param three_body_decay: exponential decay for the three body function
    :type three_body_decay: float
    :param three_body_weight: relative weight of the three body function
    :type three_body_weight: float
   '''


    nuclear_charges,coordinates = frame.get_atomic_numbers(),frame.get_positions()
    #print(nuclear_charges)
    rep = generate_fchl_acsf(nuclear_charges, coordinates, elements,
                                        nRs2=nRs2, nRs3=nRs3, nFourier=nFourier,
                                        eta2=eta2, eta3=eta3, zeta=zeta,
                                        rcut=rcut, acut=acut,
                                         two_body_decay=two_body_decay, three_body_decay=three_body_decay,
                                        three_body_weight=three_body_weight,
                                        pad=pad, gradients=False)
    #rep = generate_representation(coordinates=coordinates, nuclear_charges=nuclear_charges,max_size=5)
    return rep.flatten(), nuclear_charges
    #return np.array(rep)
    #rep_out = np.zeros((rep.shape[0],len(elements),rep.shape[1]))
    
    #for i,z in enumerate(nuclear_charges):
    #        j = np.where(np.equal(z,elements))[0][0]
    #        rep_out[i,j] = rep[i]
    #rep_out = rep_out.reshape(len(rep_out),-1)
    #return rep_out, nuclear_charges
    
def main(task=TASK_NAME, prefix= False , output= False , peratom= False, 
                    nRs2=24, nRs3=20, nFourier=1,
                    eta2=0.32, eta3=2.7, zeta=np.pi, 
                    rcut=8.0, acut=8.0,
                    two_body_decay=1.8, three_body_decay=0.57,
                    three_body_weight=13.4, 
                    periodic = False, kernel='global', n_comp=128):
    """
    Generate the FCHL19 representation (https://doi.org/10.1063/1.5126701).
    Requires the developer version of the QML package, see
    https://www.qmlcode.org/installation.html for installation instructions.
    Parameters
    ----------
    fxyz: string giving location of xyz file
    fdictxyz: string giving location of xyz file that is used as a dictionary
    prefix: string giving the filename prefix
    output: [xyz]: append the FCHL19 representation to extended xyz file; [mat] output as a standlone matrix
    :param nRs2: Number of gaussian basis functions in the two-body terms
    :type nRs2: integer
    :param nRs3: Number of gaussian basis functions in the three-body radial part
    :type nRs3: integer
    :param nFourier: Order of Fourier expansion
    :type nFourier: integer
    :param eta2: Precision in the gaussian basis functions in the two-body terms
    :type eta2: float
    :param eta3: Precision in the gaussian basis functions in the three-body radial part
    :type eta3: float
    :param zeta: Precision parameter of basis functions in the three-body angular part
    :type zeta: float
    :param two_body_decay: exponential decay for the two body function
    :type two_body_decay: float
    :param three_body_decay: exponential decay for the three body function
    :type three_body_decay: float
    :param three_body_weight: relative weight of the three body function
    :type three_body_weight: float
    :is_periodic: Boolean determining Whether the system is periodic.
    :type Boolean:
    """

    peratom = bool(peratom)
    frames = []

    # read frames
    if task != 'none':
        fxyz = 'data/'+task+'/'+task+'.xyz'
        frames = read(fxyz, ':')
        nframes = len(frames)
        print("read xyz file:", fxyz, ", a total of", nframes, "frames")

    global_species = []
    max_len = 0
    for frame in frames:
        max_len = max(max_len, len(frame.get_atomic_numbers()))
        global_species.extend(frame.get_atomic_numbers())
        if not periodic:
            frame.set_pbc([False, False, False])
    global_species = np.unique(global_species)
    print("a total of", nframes, "frames, with elements: ", global_species)
    rep, charge = repr_wrapper(frame,global_species,periodic,nRs2=nRs2, nRs3=nRs3, nFourier=nFourier,
                                                eta2=eta2, eta3=eta3, zeta=zeta, 
                                                rcut=rcut, acut=acut,
                                                two_body_decay=two_body_decay, three_body_decay=three_body_decay,
                                                three_body_weight=three_body_weight, pad = max_len)
    np_rep = np.empty((len(frames),rep.shape[0]))
    charge_list = []
    print('generating np_rep')
    for i, frame in enumerate(frames):
        #print(i,frame)
        #print(frame.get_positions())
        rep, charge = repr_wrapper(frame,global_species,periodic,nRs2=nRs2, nRs3=nRs3, nFourier=nFourier,
                                                eta2=eta2, eta3=eta3, zeta=zeta, 
                                                rcut=rcut, acut=acut,
                                                two_body_decay=two_body_decay, three_body_decay=three_body_decay,
                                                three_body_weight=three_body_weight, pad = max_len)
        np_rep[i]=rep
        charge_list.append(charge)
        #print(rep.shape)
    pca = PCA(n_comp)
    np_rep = pca.fit_transform(np_rep)
    np.save('results/FCHL/'+task+'.npy', np_rep)
    #np_rep = rep_list[0]
    #for rep in rep_list:
    #    print(rep)
    #    np_rep = np.vstack((np_rep,rep))                              
    #np_rep = np.asarray(rep_list)
    print(np_rep.shape)
    print('generating kernel')
    #for i in range(len(rep_list)):
    #    for j in range(len(rep_list)):
    #        print(get_local_kernels(rep_list[i], rep_list[j]))
    #sigmas = [2.5, 5.0, 10.0]
    #if kernel=='local':
    #   K = get_local_kernels(np_rep, np_rep, charge_list, charge_list, SIGMAS=sigmas)
    #elif kernel=='global':
    #   K = get_global_kernel(np_rep, np_rep, charge_list, charge_list, SIGMA=2.5)
    #print(K)
    #print(K.shape)
    #np.save('/rds-d2/user/wjm41/hpc-work/kernels/FCHL/'+task+'_'+kernel+'.npy',K)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, required=True, help='Task')
    parser.add_argument('--prefix', type=str, default='ASAP', help='Filename prefix')
    parser.add_argument('--output', type=str, default='xyz', help='The format for output files ([xyz], [matrix])')
                        
    parser.add_argument('--nRs2', type=int, default=24, help='number radial bins in the two-body term')
    parser.add_argument('--nRs3', type=int, default=20, help='number radial bins in the three-body term')
    parser.add_argument('--nFourier', type=int, default=1, help='Order of Fourier expansion of the angular part (Dont Change this!)')
    
    parser.add_argument('--eta2', type=float, default=0.32, help='Widths of the gaussians in the two-body term')
    parser.add_argument('--eta3', type=float, default=3.0, help='Widths of the gaussians in the three-body term')
    parser.add_argument('--zeta', type=float, default=np.pi, help='Width of the gaussian in the three-body angular part (Dont Change this!)')
    
    parser.add_argument('--rcut', type=float, default=8.0, help='Cutoff radius')
    parser.add_argument('--acut', type=float, default=6.0, help='Cutoff radius')
    
    parser.add_argument('--two_body_decay', type=float, default=1.8, help='exponent of the two-body scaling function')
    parser.add_argument('--three_body_decay', type=float, default=0.57, help='exponent of the three-body scaling function')    
    parser.add_argument('--three_body_weight', type=float, default=13.4, help='Relative weight of the three-body term')    
    
    parser.add_argument('--periodic', type=str2bool, nargs='?', const=True, default=False,
                        help='Is the system periodic (True/False)?')
    parser.add_argument('--kernel', type=str, default='global',
                        help='kernel used')
    parser.add_argument('--n_comp', type=int, default='256',
                        help='components used in PCA')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print( args)
    main(**vars(args))
