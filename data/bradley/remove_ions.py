import sys

smiles_name = sys.argv[1]+'.can'
new_smiles_name = sys.argv[1]+'_noion.can'

metal_ion_list_A = ['Li', 'Ag', 'Al', 'As', 'Au', 'Ba', 'Be', 'Bi','Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Dy', 'Fe', 'Ge', 'Hg', 'In', 'K', 'Mg', 'Mn', 'Mo', 'Nd', 'Ni', 'Pb', 'Pd', 'Pt','Sb', 'Se', 'Si', 'Sn', 'Sr', 'Ti', 'Tl', 'V', 'Yb', 'Zn', 'Zr']

metal_ion_list_B = ["Ac", "Ag", "Al", "As", "Au","Bi","Ca", "Co", "Cr", "Cs", "Cu","Fe", "Ga", "Gd", "Ge","Hg", "Ho","Ir", "K", "Li", "Mg", "Mn", "Mo","Na", "Ni","Pb", "Pd", "Pt", "Re", "Rh", "Ru", "Sb", "se", "Si", "Sn", "Tb", "Te", "Ti", "Tl", "U", "V", "W", "Zn", "Zr"]

metal_list = list(set(metal_ion_list_A + metal_ion_list_B))
#print(metal_list)
f = open(smiles_name,"r")
lines = f.readlines()
f.close()
n=0
f = open(new_smiles_name, 'w')
for line in lines:
	#print(line)
	write=True
	for i in metal_list:
		if i in line:
			write=False
			print(line)
			break
	if write:
		n+=1
		f.write(line)
print('Num Mols = '+str(n))
f.close()
