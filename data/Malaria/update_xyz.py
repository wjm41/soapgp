import pandas as pd
import sys

smiles_name = sys.argv[1]+'.can'
xyz_name = sys.argv[1]+'.xyz'
n_confs = int(sys.argv[2])

SMILES_df = pd.read_csv(smiles_name, header=None, names=['smiles','num','2','3','4','5'], delim_whitespace=True)

i=0
n=0
with open(xyz_name,'r+') as f:
    lines = f.readlines()
    f.seek(0)
    for line in lines:
        if line=='\n':
           myrow = SMILES_df.iloc[i]
           line = 'smiles="'+myrow['smiles']+'" tag="'+str(myrow['num'])+'" \n'
           n+=1
           if n%n_confs==0:
              i+=1
        f.write(line)
    f.truncate()
