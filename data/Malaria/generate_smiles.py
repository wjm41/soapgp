import pandas as pd
import sys

csv_name = sys.argv[1]+'.csv'
smiles_name = sys.argv[1]+'.can'

SMILES_df = pd.read_csv(csv_name,header=0, index_col=False)
file=open(smiles_name,'w')
for i,row in SMILES_df.iterrows():
	file.write(row['SMILES']+'\t'+str(row['Percentage_inhibition_3D7'])+'\t'+str(row['Percentage_inhibition_DD2'])+'\t'+str(row['Percentage_inhibition_3D7_PFLDH'])+'\t'+str(row['XC50_3D7 (microM)'])+'\t'+str(row['Percentage_inhibition_HEPG2'])+'\n')
file.close()
