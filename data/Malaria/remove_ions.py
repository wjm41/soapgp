import sys
import pandas as pd
smiles_name = sys.argv[1]+'.can'
new_smiles_name = sys.argv[1]+'_noion.can'
csv_name = sys.argv[1]+'.csv'
new_csv_name = sys.argv[1]+'_noion.csv'

metal_ion_list_A = ['Li', 'Ag', 'Al', 'As', 'Au', 'Ba', 'Be', 'Bi','Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Dy', 'Fe', 'Ge', 'Hg', 'In', 'K', 'Mg', 'Mn', 'Mo', 'Nd', 'Ni', 'Pb', 'Pd', 'Pt','Sb', 'Se', 'Si', 'Sn', 'Sr', 'Ti', 'Tl', 'V', 'Yb', 'Zn', 'Zr']

metal_ion_list_B = ["Ac", "Ag", "Al", "As", "Au","Bi","Ca", "Co", "Cr", "Cs", "Cu","Fe", "Ga", "Gd", "Ge","Hg", "Ho","Ir", "K", "Li", "Mg", "Mn", "Mo","Na", "Ni","Pb", "Pd", "Pt", "Re", "Rh", "Ru", "Sb", "se", "Si", "Sn", "Tb", "Te", "Ti", "Tl", "U", "V", "W", "Zn", "Zr"]

metal_list = list(set(metal_ion_list_A + metal_ion_list_B))
#print(metal_list)
f = open(smiles_name,"r")
lines = f.readlines()
f.close()
n=0
id_list = []
f = open(new_smiles_name, 'w')
for id, line in enumerate(lines):
	#print(line)
	write=True
	for i in metal_list:
		if i in line:
			write=False
			id_list.append(id)
			break
	if write:
		n+=1
		f.write(line)
print('Num Mols = '+str(n))
print('Bad mol ids =\n')
print(id_list)

df = pd.read_csv(csv_name)
df_noion = df.drop(id_list)
df_noion.to_csv(new_csv_name, index=False)

print('New csv length = '+str(len(df_noion.index)))

f.close()
