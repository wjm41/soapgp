from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent import futures
import numpy as np
import sys

max_workers=32

data = sys.argv[1]
n_confs = int(sys.argv[2])

# This function is called in the subprocess.
def generateconformations(m):
    m = Chem.AddHs(m)
    ids = AllChem.EmbedMultipleConfs(m, numConfs=n_confs)
    res = AllChem.MMFFOptimizeMoleculeConfs(m)
    energies = [i[1] for i in res]
    m = Chem.RemoveHs(m)
    return m, list(ids), energies

smiles_name = data+'.can'
energy_file = data+'_energies.txt'
sdf_name = data+'.sdf'

#n = int(sys.argv[2])

writer = Chem.SDWriter(sdf_name)

total_energy_list = []

suppl = Chem.SmilesMolSupplier(smiles_name, delimiter="\t", titleLine=False)

#file = open(energy_file,'w')

with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit a set of asynchronous jobs
    jobs = []
    for mol in suppl:
        if mol:
            job = executor.submit(generateconformations, mol)
            jobs.append(job)

    # Process the job results (in submission order) and save the conformers.
    for job in jobs:
        mol, ids, ens = job.result()
        #print(ens)
        total_energy_list.extend(ens)
        for id in ids:
            writer.write( mol, confId=id)
np.savetxt(energy_file, total_energy_list)
writer.close()
