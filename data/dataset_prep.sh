#! /bin/bash
python generate_smiles.py $1
python generate_sdf.py $1
babel -isdf $1'.sdf' -oxyz $1'.xyz'
python update_xyz.py $1 > $1'_final.xyz'
rm $1'.xyz'
mv $1'_final.xyz' $1'.xyz'
