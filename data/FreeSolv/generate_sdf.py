from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent import futures
import numpy as np
import sys

max_workers=32

data = sys.argv[1]

# This function is called in the subprocess.
def generateconformations(m):
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    m = Chem.RemoveHs(m)
    return m

smiles_name = data+'.can'
sdf_name = data+'.sdf'


writer = Chem.SDWriter(sdf_name)


suppl = Chem.SmilesMolSupplier(smiles_name, delimiter="\t", titleLine=False)


with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit a set of asynchronous jobs
    jobs = []
    for mol in suppl:
        if mol:
            job = executor.submit(generateconformations, mol)
            jobs.append(job)

    # Process the job results (in submission order) and save the conformers.
    for job in jobs:
        mol = job.result()
        writer.write( mol )
writer.close()
