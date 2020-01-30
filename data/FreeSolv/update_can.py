import pandas as pd
import sys

smiles_name = sys.argv[1]+'_multi.can'
csv_name = sys.argv[1]+'.csv'
n_conf = int(sys.argv[2])

SMILES_df = pd.read_csv(csv_name,index_col=False)
file=open(smiles_name,'w')
for i,row in SMILES_df.iterrows():
      for j in range(n_conf):
           file.write(row['smiles']+'\t'+row['iupac']+'\t'+str(row['expt'])+'\n')
file.close()
