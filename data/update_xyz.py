import pandas as pd
import sys

smiles_name = sys.argv[1]+'.can'
xyz_name = sys.argv[1]+'.xyz'

SMILES_df = pd.read_csv(smiles_name, header=None, names=['smiles','val'], delim_whitespace=True)

i=0
xyz_file = open(xyz_name,'r')

for line in xyz_file:
    if line=='\n':
        myrow = SMILES_df.iloc[i]
        line = 'smiles="'+myrow['smiles']+'" val="'+myrow['val']+'" \n'
        i+=1
    sys.stdout.write(line)
