from e3fp.config.params import read_params
from e3fp.pipeline import params_to_dicts
from e3fp.pipeline import fprints_from_smiles
from e3fp.fingerprint.db import FingerprintDatabase
from e3fp.fingerprint.fprint import Fingerprint

def gen_e3fp_features(smiles_list):
    config = read_params('single_params.cfg')
    c_params, f_params = params_to_dicts(config)
  
    fprint_list = []
    for name, smiles in enumerate(smiles_list):
        fprint_list.append(fprints_from_smiles(smiles, name, confgen_params=c_params, fprint_params=f_params)[0])
   # print(fprint_list)
    db = FingerprintDatabase(fp_type=Fingerprint, level=5)
    db.add_fingerprints(fprint_list)
    return db
