import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.preprocessing import OneHotEncoder,MultiLabelBinarizer

def get_yield_data(yield_list,task_type):
    if task_type == 'bin':
        yield_list = np.array([0 if y == 0 else 1 for y in yield_list])
    elif task_type == 'mul':
        yield_list = np.array([0 if y == 0 else 1 if y <= 10 else 2 if y <= 30 else 3 for y in yield_list])
    elif task_type == 'reg':
        yield_list = np.array([y/100 for y in yield_list])
    return yield_list

def encode_conditions(catalyst_list,base_list,solvent_list):
    # catalyst OHE
    catalyst_ohe = OneHotEncoder(handle_unknown='ignore')
    catalyst_ohe.fit(catalyst_list)

    # base OHE
    base_ohe = OneHotEncoder(handle_unknown='ignore')
    base_ohe.fit(base_list)

    # solvent MHE
    solvent_mhe = MultiLabelBinarizer()
    solvent_mhe.fit(solvent_list)
    return catalyst_ohe,base_ohe,solvent_mhe

def one_hot_encode(boronic_acid_list,aryl_halide_list):
    # boronic acid OHE
    boronic_acid_ohe = OneHotEncoder(handle_unknown='ignore')
    boronic_acid_ohe.fit(boronic_acid_list)

    # aryl halide OHE
    aryl_halide_ohe = OneHotEncoder(handle_unknown='ignore')
    aryl_halide_ohe.fit(aryl_halide_list)
    return boronic_acid_ohe,aryl_halide_ohe

def genFingerprints(smiles_list,radius,nBits): # generates Morgan fingerprints given a list of smiles strings
    fp_list = []
    for i,smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits)
        arr = np.array((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr.tolist())
    return fp_list

def get_fgp_data(boronic_acid_list,aryl_halide_list,radius,nBits):
    fp_list = []
    boronic_acid_fp_list = np.vstack(genFingerprints(boronic_acid_list,radius,nBits))
    aryl_halide_fp_list = np.vstack(genFingerprints(aryl_halide_list,radius,nBits))
    fp_list = np.hstack((boronic_acid_fp_list,aryl_halide_fp_list))
    return fp_list

def get_reactive_site_dft_data(df):
    ba_mol_list,ah_mol_list = df['BA_MoleculeFeatures'].to_list(), df['AH_MoleculeFeatures'].to_list()
    ba_boron_list,ba_carbon_list,ah_halogen_list,ah_carbon_list = df['BA_Boron_AtomFeatures'].to_list(), \
                                                                  df['BA_Carbon_AtomFeatures'].to_list(), \
                                                                  df['AH_Halogen_AtomFeatures'].to_list(), \
                                                                  df['AH_Carbon_AtomFeatures'].to_list()
    mol_desc_labels = ['dipole','molar_mass','E_scf','electronic_spatial_extent','homo_energy','lumo_energy','electronegativity','G']
    atom_desc_labels = ['VBur','Mulliken_charge','APT_charge','NPA_charge','NPA_valence','NMR_anisotropy','ES_root_Mulliken_charge','ES_root_NPA_charge']
    ba_mol_descs,ah_mol_descs,ba_boron_descs,ba_carbon_descs,ah_halogen_descs,ah_carbon_descs = [],[],[],[],[],[]

    for ba_mol,ah_mol,ba_boron,ba_carbon,ah_halogen,ah_carbon in zip(ba_mol_list,ah_mol_list,ba_boron_list,ba_carbon_list,ah_halogen_list,ah_carbon_list):
        ba_mol_descs.append([float(item) for key,item in ba_mol.items() if key in mol_desc_labels])
        ah_mol_descs.append([float(item) for key,item in ah_mol.items() if key in mol_desc_labels])
        ba_boron_descs.append([float(item) for key,item in ba_boron.items() if key in atom_desc_labels])
        ba_carbon_descs.append([float(item) for key,item in ba_carbon.items() if key in atom_desc_labels])
        ah_halogen_descs.append([float(item) for key,item in ah_halogen.items() if key in atom_desc_labels])
        ah_carbon_descs.append([float(item) for key,item in ah_carbon.items() if key in atom_desc_labels])

    dft_descriptors = np.hstack((ba_mol_descs,ah_mol_descs,ba_boron_descs,ba_carbon_descs,ah_halogen_descs,ah_carbon_descs))
    return dft_descriptors

def get_language_features(filepath):
    features = np.load(filepath)
    return features