import pandas as pd
import sys

smiles_name = sys.argv[1]+'.can'
csv_name = sys.argv[2]+'.csv'

csv = pd.read_csv(csv_name, header=0,names=['SMILES','Activity','val']) #Change to whatever the headers are in the .csv

SMILES_df = pd.read_csv(smiles_name, header=0,delim_whitespace=True,names=['SMILES','Name']) #Keep the columns you want

SMILES_df = SMILES_df.merge(csv, on='SMILES', how='left')

SMILES_df.to_csv(sys.argv[1]+'.can', header=False, index=False, columns=('SMILES','val')) #saves to .can file
