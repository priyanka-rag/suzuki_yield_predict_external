import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# SOME GLOBAL INFORMATION
boronic_acid_substruct = Chem.MolFromSmarts('[#6:2]-B([#8])[#8]')
aryl_halide_substruct = Chem.MolFromSmarts('[#6:1]-[#9,#17,#35,#53]')

def return_molecule_type(core_smi, monomer_smi):
    if (Chem.MolFromSmiles(core_smi).HasSubstructMatch(boronic_acid_substruct)): 
        boronic_acid, aryl_halide = core_smi, monomer_smi
    elif (Chem.MolFromSmiles(monomer_smi).HasSubstructMatch(boronic_acid_substruct)): 
        boronic_acid, aryl_halide = monomer_smi, core_smi
    return boronic_acid, aryl_halide

def atommap_reaction(core, monomer, obs_product):
    suzuki_template = "[#6:2]-B([#8])[#8].[#6:1]-[#9,#17,#35,#53]>>[*:1]-[*:2]"
    suzuki_rxn = AllChem.ReactionFromSmarts(suzuki_template)

    boronic_acid, aryl_halide = return_molecule_type(core, monomer)
    reactants = (Chem.MolFromSmiles(boronic_acid),Chem.MolFromSmiles(aryl_halide))
    obs_product = Chem.MolFromSmiles(obs_product)

    unmapped = 1
    for rct in reactants:
        for a in rct.GetAtoms():
            #if not a.HasProp('molAtomMapNumber'): a.SetIntProp('molAtomMapNumber', unmapped)
            a.SetIsotope(unmapped)
            unmapped += 1
    products = suzuki_rxn.RunReactants((reactants[0],reactants[1]))

    #check if any of the template products match the observed one
    try:
        for prod in products:
            prod_smi = Chem.MolToSmiles(prod[0])
            copy_prod = prod[0]
            [a.SetIsotope(0) for a in copy_prod.GetAtoms()]
            if (copy_prod.HasSubstructMatch(obs_product) and obs_product.HasSubstructMatch(copy_prod)): 
                matched_product = prod_smi
        matched_product = Chem.MolFromSmiles(matched_product)
        # Replace isotope mapping with atom mapping
        [a.SetAtomMapNum(a.GetIsotope()) for a in reactants[0].GetAtoms()]
        [a.SetAtomMapNum(a.GetIsotope()) for a in reactants[1].GetAtoms()]
        [a.SetAtomMapNum(a.GetIsotope()) for a in matched_product.GetAtoms()]
        [a.SetIsotope(0) for a in reactants[0].GetAtoms()]
        [a.SetIsotope(0) for a in reactants[1].GetAtoms()]
        [a.SetIsotope(0) for a in matched_product.GetAtoms()]
        matched_product_smi = Chem.MolToSmiles(matched_product)
    except: #if none match
        matched_product_smi = 'ERROR'
    
    atommapped_rxn = Chem.MolToSmiles(reactants[0]) + '.' + Chem.MolToSmiles(reactants[1]) + '>>' + matched_product_smi
    return atommapped_rxn

def add_atommapping_to_data(df):
    mapped_rxns_list = []
    for i,row in df.iterrows():
        core, monomer, obs_product = row['Core_Smiles'], row['Monomer_Smiles'], row['Product_Smiles']
        mapped_rxns_list.append(atommap_reaction(core, monomer, obs_product))
    
    assert(len(mapped_rxns_list) == df.shape[0]) #dummy check
    df['Atom_Mapped_Reaction'] = mapped_rxns_list()
    return df