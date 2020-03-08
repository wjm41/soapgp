#! /bin/bash
python generate_smiles.py $1  #Generates .can file
python generate_sdf.py $1 #generates .sdf file using rdkit conformer generation
babel -isdf $1'.sdf' -oxyz $1'.xyz' #converts to .xyz
python update_xyz.py $1 > $1'_final.xyz' #updates to include smiles names 
rm $1'.xyz'
mv $1'_final.xyz' $1'.xyz'
