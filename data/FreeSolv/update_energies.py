import numpy as np
import sys

num_conf = int(sys.argv[1]) 

factor = 1000/(6.02E23*1.38E-23*298)

energies = np.loadtxt('FreeSolv_energies.txt')

split_energies = np.array(np.split(energies,len(energies)/num_conf)) 

split_energies = np.exp(-1*split_energies*factor)

sums = np.sum(split_energies, axis=1)
sums = np.array(np.split(sums,len(energies)/num_conf))
sums = np.repeat(sums, num_conf, axis=1)
#print(sums)

conf_weight = split_energies/sums
#print(conf_weight)

np.savetxt('FreeSolv_weights.txt',conf_weight)
