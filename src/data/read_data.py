import pandas as pd
from ast import literal_eval

# some general converters to help read in various datasets
converters = {'Catalyst': pd.eval, 'Base': pd.eval, 'Ligand': pd.eval, 'Solvent': pd.eval, 'Solvent_Unsplit': pd.eval,
              'BA_Boron_AtomFeatures': literal_eval, 'BA_Carbon_AtomFeatures': literal_eval, 'AH_Halogen_AtomFeatures': literal_eval, 'AH_Carbon_AtomFeatures': literal_eval,
              'BA_MoleculeFeatures': literal_eval, 'AH_MoleculeFeatures': literal_eval}

# define main dataset directory
main_datasets_dir = './dataset_files'

def read_retrospective_data():
    df = pd.read_csv(f'{main_datasets_dir}/retrospective/suzuki_data.csv',  
                     converters=converters)
    return df

def read_post_2021_data():
    df = pd.read_csv(f'{main_datasets_dir}/post-2021/post_2021_suzuki_data.csv',
                       converters=converters)
    return df

def read_monomer_replacement_data():
    df = pd.read_csv(f'{main_datasets_dir}/monomer_replacement/replacement_reactions_enumerated.csv',  
                     converters=converters)
    return df