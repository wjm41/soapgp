import sys

smiles_name = sys.argv[1]+'.can'
new_smiles_name = sys.argv[1]+'_noion.can'

metal_ion_list= ['Li', 'Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi','Ca', 'Cd', 'Co', 'Cr', 'Cu', 'Dy', 'Fe', 'Ge', 'Hg', 'I', 'In', 'K', 'Mg', 'Mn', 'Mo', 'Nd', 'Ni', 'Pb', 'Pd', 'Pt','Sb', 'Se', 'Si', 'Sn', 'Sr', 'Ti', 'Tl', 'V', 'Yb', 'Zn', 'Zr']

metal_ion_list = ["Ac", "Ag", "Al", "As", "Au","Bi","Ca", "Co", "Cr", "Cs", "Cu","Fe", "Ga", "Gd", "Ge","Hg", "Ho","Ir", "K", "Li", "Mg", "Mn", "Mo","Na", "Ni","Pb", "Pd", "Pt", "Re", "Rh", "Ru", "Sb", "Se", "Si", "Sn", "Tb", "Te", "Ti", "Tl", "U", "V", "W", "Zn", "Zr"]

f = open(smiles_name,"r")
lines = f.readlines()

f.close()
f = open(new_smiles_name, 'w')
for line in lines:
	write=True
	for i in metal_ion_list:
		if i in line:
			write=False
			break
	if write:
		f.write(line)
f.close()
