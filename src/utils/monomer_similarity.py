import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from espsim import EmbedAlignConstrainedScore, EmbedAlignScore, ConstrainedEmbedMultipleConfs, GetEspSim, GetShapeSim
from espsim.helpers import mlCharges

ba_mol = Chem.MolFromSmarts('[#5]([#8H])[#8H]')

def replace(smi):
    start_mol = Chem.MolFromSmiles(smi)
    mod_mol = Chem.ReplaceSubstructs(start_mol, Chem.MolFromSmiles('B(OC)OC'), Chem.MolFromSmiles('B(O)O'), replaceAll=True)
    return Chem.MolToSmiles(mod_mol[0])

def keep_largest_frag(smi):
    split_smi_list = smi.split('.')
    split_smi_list.sort(key=lambda x: Chem.Descriptors.MolWt(Chem.MolFromSmiles(x)), reverse=True)
    smi = split_smi_list[0]
    return smi

#converts all organoborane monomers to boronic acids for the purpose of monomer similarity calculations
def convert_bpin_to_bacid(smi_list):
    mod_smi_list = [keep_largest_frag(replace(smi)) for smi in smi_list]
    #sanity check
    valid_smi_list = [smi for smi in mod_smi_list if Chem.MolFromSmiles(smi).HasSubstructMatch(ba_mol)]
    return valid_smi_list

#calculates the similarity score between 2 molecules, on the basis of shape and electrostatics
def calc_similarity(mol1_smi, mol2_smi):
    #keep mol1 as the probe, mol2 as the reference
    prbSmiles, refSmiles = mol1_smi, mol2_smi
    prbMol = Chem.AddHs(Chem.MolFromSmiles(prbSmiles))
    refMol = Chem.AddHs(Chem.MolFromSmiles(refSmiles))

    try:
        simShape, simEsp = EmbedAlignConstrainedScore(prbMol, refMol, core, renormalize=True,
                                                    prbNumConfs=50, refNumConfs=50, metric='carbo',
                                                    integrate='gauss', partialCharges='ml', getBestESP=False)

        simShape, simEsp = simShape[0], simEsp[0]
        sim_score = simShape * simEsp
    except:
        simShape = np.nan
        simEsp = np.nan
        sim_score = np.nan
    return simShape, simEsp, sim_score

#define a helper molecule, some boronic acid from the dataset
helper = Chem.AddHs(Chem.MolFromSmiles('OB(O)c1cccnc1'))
AllChem.EmbedMolecule(helper, AllChem.ETKDG())  #embed reference molecule, create one conformer
AllChem.UFFOptimizeMolecule(helper)  #optimize the coordinates of the conformer
core = AllChem.DeleteSubstructs(AllChem.ReplaceSidechains(helper, ba_mol),
                                Chem.MolFromSmiles('*'))  #Create core molecule with 3D coordinates
core.UpdatePropertyCache()

mol1_smi = 'OB(O)c1ccc(N2CCOCC2)cc1'
mol2_smi = 'CN1CCN(c2ccc(B(O)O)cc2)CC1'
print(calc_similarity(mol1_smi, mol2_smi))