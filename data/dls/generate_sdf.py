from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent import futures
import progressbar
import sys

max_workers=32

# This function is called in the subprocess.
def generateconformations(m):
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    m = Chem.RemoveHs(m)
    return m

data = sys.argv[1]
smiles_name = data+'.can'
sdf_name = data+'.sdf'

#n = int(sys.argv[2])

writer = Chem.SDWriter(sdf_name)

suppl = Chem.SmilesMolSupplier(smiles_name, delimiter="\t", titleLine=False)

with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Submit a set of asynchronous jobs
    jobs = []
    for mol in suppl:
        if mol:
            job = executor.submit(generateconformations, mol)
            jobs.append(job)

    widgets = ["Generating conformations; ", progressbar.Percentage(), " ",
               progressbar.ETA(), " ", progressbar.Bar()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(jobs))

    # Process the job results (in submission order) and save the conformers.
    for job in jobs:
        mol= job.result()
        writer.write(mol)
        
writer.close()
