from autoqchem_boosted.imports import *
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

workflow_type="equilibrium"
theory="APFD"
solvent="None"
light_basis_set="6-31G*"
heavy_basis_set="LANL2DZ"
generic_basis_set="genecp"
max_light_atomic_number=36
num_conf = 20

# Define list of molecules and IDs
unique_molecules_df = pd.read_csv(f'{main_datasets_dir}/retrospective/suzuki_dataset_unique_molecules_fragments_removed.csv')
smiles_list = unique_molecules_df['SMILES'].to_list()
molecule_ids = unique_molecules_df['Molecule ID'].to_list()

for smiles_str,molecule_id in zip(smiles_list,molecule_ids):
    # create internal rdkit mol object just for some later things
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles_str))

    # initialize obmol
    obmol = pybel.readstring('smi', smiles_str).OBMol
    obmol.AddHydrogens()

    # initial geometry
    gen3D = pybel.ob.OBOp.FindType("gen3D")
    ob_gen3D_option = 'best'
    gen3D.Do(obmol, ob_gen3D_option)

    # conf search
    confSearch = pybel.ob.OBConformerSearch()
    confSearch.Setup(obmol, num_conf)
    confSearch.Search()
    confSearch.GetConformers(obmol)

    py_mol = pybel.Molecule(obmol)
    elements = [GetSymbol(atom.atomicnum) for atom in py_mol.atoms]
    charges = np.array([atom.formalcharge for atom in py_mol.atoms])

    n_atoms = len(py_mol.atoms)
    connectivity_matrix = np.zeros((n_atoms, n_atoms))
    for bond in pybel.ob.OBMolBondIter(obmol):
        i = bond.GetBeginAtomIdx() - 1
        j = bond.GetEndAtomIdx() - 1
        bo = bond.GetBondOrder()
        connectivity_matrix[i, j] = bo
        connectivity_matrix[j, i] = bo

    # Retrieve conformer coordinates and energies
    ob_conformer_coordinates = []
    ob_conformer_energies = []
    for i in range(obmol.NumConformers()):
        obmol.SetConformer(i)
        ob_conformer_energies.append(obmol.GetEnergy(i))
        coordinates = np.array([atom.coords for atom in py_mol.atoms])
        ob_conformer_coordinates.append(coordinates)
    ob_conformer_coordinates = np.array(ob_conformer_coordinates)

    conformer_energies = []
    conformer_coordinates = []
    for cid,conformer in enumerate(ob_conformer_coordinates):
        #faulthandler.enable()
        coordinates = conformer
        try:
            atomic_input = qcel.models.AtomicInput(
                molecule = qcel.models.Molecule(
                    symbols = elements,
                    geometry = coordinates.flatten().tolist()
                ),
                driver = "energy",
                model = {
                    "method": "GFN2-xTB",
                },
                keywords = {
                    "accuracy": 1.0,
                    "max_iterations": 100,
                },
            )
        except: 
            with open(f'{directory}/conformer_distancetooclose_molecules.txt', 'a') as f:
                f.write(molecule_id + '\n')
            continue
        atomic_result = run_qcschema(atomic_input)
        if atomic_result.success: conformer_energies.append((atomic_result.return_result,conformer))

    if len(conformer_energies) == 0: #no valid conformers generated through xtb 
        #print('No valid conformers generated for ' + molecule_id)
        conformer_energies = [(ob_conformer_energies[i],ob_conformer_coordinates[i]) for i in range(obmol.NumConformers())] # use the OpenBabel-calculated energies
        #print(conformer_energies)

    conformer_energies = sorted(conformer_energies, key = lambda x: x[0])
    try:
        (e, conformer) = conformer_energies.pop(0) #extract lowest-energy conformer
    except:
        with open(f'{directory}/conformer_noopenbabelnoxtb_molecules.txt', 'a') as f:
            f.write(molecule_id + '\n')
    conformer_coordinates.append(conformer)
    conformer_coordinates = np.array(conformer_coordinates)

    can = smiles_str
    inchi = Chem.MolToInchi(rdmol)
    inchikey = Chem.MolToInchiKey(rdmol)

    # add configuration info
    max_num_conformers = num_conf
    conformer_engine = "openbabel"

    # add charge and spin
    charge = sum(charges)
    spin = Chem.Descriptors.NumRadicalElectrons(rdmol) + 1

    #pdb.set_trace()
    molecule_workdir = os.path.join(directory, "inputfiles", molecule_id)

    # generate input gaussian files
    atomic_nums = set(atom.GetAtomicNum() for atom in rdmol.GetAtoms())
    light_elements = [GetSymbol(n) for n in atomic_nums if n <= max_light_atomic_number]
    heavy_elements = [GetSymbol(n) for n in atomic_nums if n > max_light_atomic_number]
    #pdb.set_trace()
    heavy_block = ""

    if heavy_elements:
        basis_set = generic_basis_set
        heavy_block += f"{' '.join(light_elements + ['0'])}\n"
        heavy_block += f"{light_basis_set}\n****\n"
        heavy_block += f"{' '.join(heavy_elements + ['0'])}\n"
        heavy_block += f"{heavy_basis_set}\n****\n"
        heavy_block += f"\n"
        heavy_block += f"{' '.join(heavy_elements + ['0'])}\n"
        heavy_block += f"{heavy_basis_set}\n"
    else:
        basis_set = light_basis_set

    #pdb.set_trace()
    solvent_input = f"SCRF=(Solvent={solvent}) " if solvent.lower() != "none" else ""

    if workflow_type == "equilibrium":
        tasks = (
            f"opt=CalcFc {theory}/{basis_set} {solvent_input}scf=xqc ",
            f"freq {theory}/{basis_set} {solvent_input}volume NMR pop=NPA density=current Geom=AllCheck Guess=Read",
            f"TD(NStates=10, Root=1) {theory}/{basis_set} {solvent_input}volume pop=NPA density=current Geom=AllCheck Guess=Read"
        )

    gaussian_config = {'theory': theory,
                        'solvent': solvent,
                        'light_basis_set': light_basis_set,
                        'heavy_basis_set': heavy_basis_set,
                        'generic_basis_set': generic_basis_set,
                        'max_light_atomic_number': max_light_atomic_number}
    
    os.makedirs(molecule_workdir, exist_ok=True)

    # resources configuration
    n_processors = max(1, min(config['remote']['max_processors'],
                                rdmol.GetNumAtoms() // config['remote']['atoms_per_processor']))
    ram = n_processors * config['remote']['ram_per_processor']
    resource_block = f"%nprocshared={n_processors}\n%Mem={ram}GB\n"

    logger.info(f"Generating Gaussian input files for {obmol.NumConformers()} conformations for {molecule_id}.")

    for conf_id, conf_coord in enumerate(conformer_coordinates):
        # set conformer
        #conf_name = f"{self.molecule.can}_conf_{conf_id}"
        conf_name = f"conf_{conf_id}"

        # coordinates block
        geom_np_array = np.concatenate((np.array([elements]).T, conf_coord), axis=1)
        coords_block = "\n".join(map(" ".join, geom_np_array))
    
    gau_output = ""

    # loop through the tasks in the workflow and create input file
    for i, task in enumerate(tasks):
        if i == 0:  # first task is special, coordinates follow
            gau_output += resource_block
            gau_output += f"%Chk={molecule_id}_{conf_name}_{i}.chk\n"
            gau_output += f"# {task}\n\n"
            gau_output += f"{conf_name}\n\n"
            gau_output += f"{charge} {spin}\n"
            gau_output += f"{coords_block.strip()}\n"
            gau_output += f"\n"
        else:
            gau_output += "\n--Link1--\n"
            gau_output += resource_block
            gau_output += f"%Oldchk={molecule_id}_{conf_name}_{i - 1}.chk\n"
            gau_output += f"%Chk={molecule_id}_{conf_name}_{i}.chk\n"
            gau_output += f"# {task}\n"
            gau_output += f"\n"

        gau_output += heavy_block  # this is an empty string if no heavy elements are in the molecule

    gau_output += f"\n\n"

    file_path = f"{molecule_workdir}/{conf_name}.gjf"
    with open(file_path, "w") as file:
        file.write(gau_output)

    # create corresponding shell scripts for the gaussian jobs and run them
    for gjf_file in glob.glob(f"{molecule_workdir}/*.gjf"):
        base_name = os.path.join(molecule_id,os.path.basename(os.path.splitext(gjf_file)[0]))

        with open(f"{directory}/inputfiles/{base_name}.gjf") as f:
            file_string = f.read()

        n_processors = re.search("nprocshared=(.*?)\n", file_string).group(1)
        
        #create shell script for this molecule & conformer
        sh_output = ""
        sh_output += f"#{can}\n"
        sh_output +=  f"#!/bin/tcsh\n" \
                f"#BSUB -n {n_processors}\n" \
                f"#BSUB -R 'span[hosts=1]'\n" \
                f"#BSUB -R 'select[avx]'\n" \
                f"#BSUB -x\n" \
                f"#BSUB -o out.%J\n" \
                f"#BSUB -e err.%J\n\n"
        sh_output += f"export PGI_FASTMATH_CPU=haswell\n" \
                f"export g16root={directory}\n" \
                f"export PATH=$g16root/g16/:$g16root/gv:$PATH\n\n"
        sh_output += f"export GAUSS_SCRDIR=./scratch\n" \
                f". $g16root/g16/bsd/g16.profile\n" \
                f"mkdir -p $GAUSS_SCRDIR\n" \
                f"chmod 750 $GAUSS_SCRDIR\n\n"
        sh_output += f"input='{base_name}'\n\n"
        sh_output += f"echo 'running Gaussian for {base_name}:' $(date)\n" \
                f"cd {directory}\n" \
                f"g16 < {directory}/inputfiles/${{input}}.gjf > {directory}/outputfiles/${{input}}.log\n\n"
        sh_output += f"printf '{base_name}\\n' >> {record_directory}\n"
        sh_output += f"echo 'done with Gaussian for {base_name}:' $(date)\n"

        os.makedirs(os.path.join(directory,'outputfiles',molecule_id), exist_ok=True) #make directory in outputfiles for this molecule
        sh_file_path = f"{directory}/shfiles_test/{molecule_id}.sh"
        sh_file_path_tocall = sh_file_path.replace('(',f'\(').replace(')',f'\)')
        with open(sh_file_path, "w") as f:
            f.write(sh_output)
        logger.debug(f"Created a job file in {sh_file_path}")
