import json
from collections import namedtuple
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from data.read_data import *
from utils.atom_mapping import *

# get the molecule ID of a molecule given its SMILES
def smiles_to_mol_id(smi):
    # read df of smiles-molecule_id mapping
    smiles_id_df = pd.read_csv(f'{main_datasets_dir}/retrospective/suzuki_dataset_unique_molecules_fragments_removed.csv')
    mol_id = smiles_id_df.loc[smiles_id_df['Can_SMILES'] == smi, 'Molecule ID'].iloc[0]
    return mol_id 

# read .json file of DFT descriptors for a certain molecule_id
def read_json_file(molecule_id):
    desc_filepath = f"{main_datasets_dir}/molecule_descriptors/{molecule_id}.json"
    with open(desc_filepath) as project_file:    
        dft_descriptors = json.load(project_file)
    return dft_descriptors

# credit for this function: https://greglandrum.github.io/rdkit-blog/posts/2021-11-26-highlighting-changed-bonds-in-reactions.html
def map_reacting_atoms_to_products(rxn,reactingAtoms,AtomInfo):
    ''' figures out which atoms in the products each mappd atom in the reactants maps to '''
    res = []
    for ridx,reacting in enumerate(reactingAtoms):
        reactant = rxn.GetReactantTemplate(ridx)
        for raidx in reacting:
            mapnum = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
            foundit=False
            for pidx,product in enumerate(rxn.GetProducts()):
                for paidx,patom in enumerate(product.GetAtoms()):
                    if patom.GetAtomMapNum()==mapnum:
                        res.append(AtomInfo(mapnum,ridx,raidx,pidx,paidx))
                        foundit = True
                        break
                    if foundit:
                        break
    return res

# credit for this function: https://greglandrum.github.io/rdkit-blog/posts/2021-11-26-highlighting-changed-bonds-in-reactions.html
def get_mapped_neighbors(atom):
    ''' test all mapped neighbors of a mapped atom'''
    res = {}
    amap = atom.GetAtomMapNum()
    if not amap:
        return res
    for nbr in atom.GetNeighbors():
        nmap = nbr.GetAtomMapNum()
        if nmap:
            if amap>nmap:
                res[(nmap,amap)] = (atom.GetIdx(),nbr.GetIdx())
            else:
                res[(amap,nmap)] = (nbr.GetIdx(),atom.GetIdx())
    return res

# credit for this function: https://greglandrum.github.io/rdkit-blog/posts/2021-11-26-highlighting-changed-bonds-in-reactions.html
def find_modifications_in_product(rxn,AtomInfo,BondInfo):
    ''' returns a 2-tuple with the modified atoms and bonds from the reaction '''
    reactingAtoms = rxn.GetReactingAtoms()
    amap = map_reacting_atoms_to_products(rxn,reactingAtoms,AtomInfo)
    res = []
    seen = set()
    # this is all driven from the list of reacting atoms:
    for _,ridx,raidx,pidx,paidx in amap:
        reactant = rxn.GetReactantTemplate(ridx)
        ratom = reactant.GetAtomWithIdx(raidx)
        product = rxn.GetProductTemplate(pidx)
        patom = product.GetAtomWithIdx(paidx)

        rnbrs = get_mapped_neighbors(ratom)
        pnbrs = get_mapped_neighbors(patom)
        for tpl in pnbrs:
            pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
            if (pidx,pbond.GetIdx()) in seen:
                continue
            seen.add((pidx,pbond.GetIdx()))
            if not tpl in rnbrs:
                # new bond in product
                res.append(BondInfo(pidx,pnbrs[tpl],pbond.GetIdx(),'New'))
            else:
                # present in both reactants and products, check to see if it changed
                rbond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
                if rbond.GetBondType()!=pbond.GetBondType():
                    res.append(BondInfo(pidx,pnbrs[tpl],pbond.GetIdx(),'Changed'))
    return amap,res

# finds the reactive sites on the boronic acid and aryl halide given a mapped reaction
def find_reactive_atoms(mapped_rxn):
    # define namedtuples for storing changed atom(s) and formed bond(s) info
    AtomInfo = namedtuple('AtomInfo',('mapnum','reactant','reactantAtom','product','productAtom'))
    BondInfo = namedtuple('BondInfo',('product','productAtoms','productBond','status'))

    # initialize reaction
    mapped_rxn = AllChem.ReactionFromSmarts(mapped_rxn)
    mapped_rxn.Initialize()
    reactants = mapped_rxn.GetReactants()

    # now find changed atoms and formed bonds
    atoms, bonds = find_modifications_in_product(mapped_rxn,AtomInfo,BondInfo)

    # find the atom-mapped indices (from the original reaction) of the reactive carbons, as given by the formed bond
    involved_product_atoms = getattr(bonds[0],'productAtoms')
    involved_reactant_carbons_product_mapnums, involved_reactant_carbons_reactant_mapnums = [], []
    for product_atom in involved_product_atoms:
        for atom in atoms:
            if getattr(atom,'productAtom') == product_atom: 
                involved_reactant_carbons_product_mapnums.append(getattr(atom,'mapnum')) #overall atom mapping as given in the context of the full mapped reaction
                involved_reactant_carbons_reactant_mapnums.append(getattr(atom,'reactantAtom')) #relative atom mapping for each reactant

    # reorder the mapping numbers so they are in the order of [boronic acid, aryl halide] for everything
    if involved_reactant_carbons_product_mapnums[0] > involved_reactant_carbons_product_mapnums[1]: 
        involved_reactant_carbons_reactant_mapnums.reverse()

    boronic_acid_reactivecarbon = min(involved_reactant_carbons_product_mapnums)
    aryl_halide_reactivecarbon = max(involved_reactant_carbons_product_mapnums)
    boronic_acid_relative_reactivecarbon = involved_reactant_carbons_reactant_mapnums[0]
    aryl_halide_relative_reactivecarbon = involved_reactant_carbons_reactant_mapnums[1]

    # now find the atom-mapped indices (from the original reaction) of the reactive boron and halogen
    # bit of hard coding of atom numbers here, specific to Suzuki reactions. Change for a different reaction type
    boronic_acid, aryl_halide = reactants[0], reactants[1]
    ba_reactivecarbonatom_neighbors = boronic_acid.GetAtomWithIdx(boronic_acid_relative_reactivecarbon).GetNeighbors()
    ah_reactivecarbonatom_neighbors = aryl_halide.GetAtomWithIdx(aryl_halide_relative_reactivecarbon).GetNeighbors()
    for atom in ba_reactivecarbonatom_neighbors:
        if atom.GetAtomicNum() == 5: boronic_acid_reactiveboron = atom.GetAtomMapNum() #if atom is a boron
    for atom in ah_reactivecarbonatom_neighbors:
        if atom.GetAtomicNum() in [9,17,35,53]: aryl_halide_reactivehalogen = atom.GetAtomMapNum() #if atom is a halogen
    return boronic_acid_reactiveboron, boronic_acid_reactivecarbon, aryl_halide_reactivehalogen, aryl_halide_reactivecarbon

# extracts all molecule-level descriptors from the .json file associated with this molecule
def extract_molecule_descriptors(core_smi, monomer_smi):
   # identify the boronic acid and aryl halide, and get their molecule IDs
    boronic_acid, aryl_halide = return_molecule_type(core_smi,monomer_smi)
    boronic_acid_molid = smiles_to_mol_id(boronic_acid)
    aryl_halide_molid = smiles_to_mol_id(aryl_halide)

    # read corresponding .json file
    boronic_acid_descriptors = read_json_file(boronic_acid_molid)
    aryl_halide_descriptors = read_json_file(aryl_halide_molid)

    # read molecule-level features from "descriptors" 
    ba_molecule_descriptors = boronic_acid_descriptors['descriptors']
    ah_molecule_descriptors = aryl_halide_descriptors['descriptors']
    return ba_molecule_descriptors, ah_molecule_descriptors

# extracts all atom-level (reactive site specific) descriptors for the molecules in a reaction from the .json files associated with these molecules
def extract_atom_descriptors(core_smi,monomer_smi,obs_product_smi):
   # identify the boronic acid and aryl halide, and get their molecule IDs
    boronic_acid, aryl_halide = return_molecule_type(core_smi,monomer_smi)
    boronic_acid_molid = smiles_to_mol_id(boronic_acid)
    aryl_halide_molid = smiles_to_mol_id(aryl_halide)

    # get reactive atoms from this reaction
    mapped_rxn = atommap_reaction(boronic_acid,aryl_halide,obs_product_smi)
    boronic_acid_reactiveboron, boronic_acid_reactivecarbon, aryl_halide_reactivehalogen, aryl_halide_reactivecarbon = find_reactive_atoms(mapped_rxn)

    # get atom-level (reactive site specific) descriptors, using the atom mapping, for the boronic acid and aryl halide
    ba_dft_descriptors = read_json_file(boronic_acid_molid)
    ah_dft_descriptors = read_json_file(aryl_halide_molid)
    ba_atom_descriptors, ah_atom_descriptors = ba_dft_descriptors['atom_descriptors'], ah_dft_descriptors['atom_descriptors']
    boronic_acid_boronidx = ba_dft_descriptors['atommaps'].index(boronic_acid_reactiveboron)
    boronic_acid_carbonidx = ba_dft_descriptors['atommaps'].index(boronic_acid_reactivecarbon)
    aryl_halide_halogenidx = [i+len(ba_dft_descriptors['atommaps']) for i in ah_dft_descriptors['atommaps']].index(aryl_halide_reactivehalogen)
    aryl_halide_carbonidx = [i+len(ba_dft_descriptors['atommaps']) for i in ah_dft_descriptors['atommaps']].index(aryl_halide_reactivecarbon)

    # sanity check that we got the right atoms
    atom1_label, atom2_label, atom3_label, atom4_label = ba_dft_descriptors['labels'][boronic_acid_boronidx], ba_dft_descriptors['labels'][boronic_acid_carbonidx], ah_dft_descriptors['labels'][aryl_halide_halogenidx], ah_dft_descriptors['labels'][aryl_halide_carbonidx]
    if atom1_label == 'B' and atom2_label == 'C' and atom3_label in ['F','Cl','Br','I'] and atom4_label == 'C':
        boronic_acid_boronfeatures = dict([(key,value[boronic_acid_boronidx]) for key,value in ba_atom_descriptors.items()])
        boronic_acid_carbonfeatures = dict([(key,value[boronic_acid_carbonidx]) for key,value in ba_atom_descriptors.items()])
        aryl_halide_halogenfeatures = dict([(key,value[aryl_halide_halogenidx]) for key,value in ah_atom_descriptors.items()])
        aryl_halide_carbonfeatures = dict([(key,value[aryl_halide_carbonidx]) for key,value in ah_atom_descriptors.items()])
    else: print(f'False for {core_smi}, {monomer_smi}')
    return boronic_acid_boronfeatures, boronic_acid_carbonfeatures, aryl_halide_halogenfeatures, aryl_halide_carbonfeatures, mapped_rxn

# adds columns of molecule-level and atom-level, reactive site DFT descriptors to a dataset df
def extract_descriptors_all_reactions(df):
    failed_reactions = [] #keep track of reactions that couldn't be atom-mapped, or involved molecules without computed descriptors
    ba_molfeatures_list, ah_molfeatures_list, ba_boronfeatures_list, ba_carbonfeatures_list, ah_halogenfeatures_list, ah_carbonfeatures_list, mapped_rxns_list = [],[],[],[],[],[],[]

    for i,row in df.iterrows():
        try:
            core_smi, monomer_smi, obs_product_smi = row['Core_Smiles'], row['Monomer_Smiles'], row['Product_Smiles']
            # get the molecule-level features
            boronic_acid_moleculefeatures, aryl_halide_moleculefeatures = extract_molecule_descriptors(core_smi,monomer_smi)
            ba_molfeatures_list.append(boronic_acid_moleculefeatures)
            ah_molfeatures_list.append(aryl_halide_moleculefeatures)

            # get the atom-level features
            boronic_acid_boronfeatures, boronic_acid_carbonfeatures, aryl_halide_halogenfeatures, aryl_halide_carbonfeatures, mapped_rxn = extract_atom_descriptors(core_smi,monomer_smi,obs_product_smi)
            ba_boronfeatures_list.append(boronic_acid_boronfeatures)
            ba_carbonfeatures_list.append(boronic_acid_carbonfeatures)
            ah_halogenfeatures_list.append(aryl_halide_halogenfeatures)
            ah_carbonfeatures_list.append(aryl_halide_carbonfeatures)
            mapped_rxns_list.append(mapped_rxn)
        except: failed_reactions.append(i)

    df.drop(index=failed_reactions,inplace=True)
    df['Atom_Mapped_Reaction'] = mapped_rxns_list
    df['BA_MoleculeFeatures'] = ba_molfeatures_list
    df['AH_MoleculeFeatures'] = ah_molfeatures_list
    df['BA_Boron_AtomFeatures'] = ba_boronfeatures_list
    df['BA_Carbon_AtomFeatures'] = ba_carbonfeatures_list
    df['AH_Halogen_AtomFeatures'] = ah_halogenfeatures_list
    df['AH_Carbon_AtomFeatures'] = ah_carbonfeatures_list
    return df
