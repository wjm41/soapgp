import pandas as pd
import sys

csv_name = sys.argv[1]+'.csv'
smiles_name = sys.argv[1]+'.can'

csv = pd.read_csv(csv_name, header=0,names=['smiles','Activity','val']) #Change to whatever the headers are in the .csv


#writes to .can
file=open(smiles_name,'w')
for i,row in csv.iterrows():
	file.write(row['smiles']+'\t'+row['val']+'\n')
file.close()
