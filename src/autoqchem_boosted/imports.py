try:
    from openbabel import pybel

    GetSymbol = pybel.ob.GetSymbol
    GetVdwRad = pybel.ob.GetVdwRad
except ImportError:
    import pybel

    table = pybel.ob.OBElementTable()
    GetSymbol = table.GetSymbol
    GetVdwRad = table.GetVdwRad
conv = pybel.ob.OBConversion()

import numpy as np
import re
import os
import itertools
from collections import Counter
import glob
import logging
import pandas as pd

from rdkit import Chem, Geometry
from rdkit.Chem import AllChem
from xtb.qcschema.harness import run_qcschema
import qcelemental as qcel

from data.read_data import *

import yaml
config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yml")))

#DEFINE DIRECTORIES
directory = f"{main_datasets_dir}/gaussian" #working data directory for saving gaussian input files, scripts, and output files
record_directory = f"{directory}/gaussian_done_molecules.txt" #where to record fully done molecules