import pandas as pd
import sys

smiles_name = sys.argv[1]+'.can'
xyz_name = sys.argv[1]+'.xyz'

SMILES_df = pd.read_csv(smiles_name, header=None, names=['smiles','num'], delim_whitespace=True)

n=0
with open(xyz_name,'r+') as f:
    lines = f.readlines()
    f.seek(0)
    for line in lines:
        if line=='\n':
           myrow = SMILES_df.iloc[n]
           line = 'smiles="'+myrow['smiles']+'" tag="'+myrow['num']+'" \n'
           n+=1
        f.write(line)
    f.truncate()
