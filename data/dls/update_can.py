import pandas as pd
import sys

csv_name = sys.argv[1]+'.csv'
smiles_name = sys.argv[1]+'.can'

SMILES_df = pd.read_csv(csv_name,header=0, index_col=False)
file=open(smiles_name,'w')
for i,row in SMILES_df.iterrows():
	file.write(row['SMILES']+'\t'+row['Chemical name']+'\t'+str(row['LogS exp (mol/L)'])+'\n')
file.close()
