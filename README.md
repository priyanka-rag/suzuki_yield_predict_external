# suzuki_yield_predict: Library-Based Suzuki Yield Prediction leveraging AbbVie's 15-year Historical Library Dataset

This repository provides code for [insert paper name and link] and the associated Supplementary Information [insert link]

_Code Author: Priyanka Raghavan_

## Install and setup

After git cloning the repository, please run the following to build the environment and extract the dataset files.

```
conda env create -f environment.yml
conda activate suzuki_env
tar -xf dataset_files.tar.gz
python setup.py develop
```

This environment was tested on Ubuntu 20.04.5 with CUDA Version 11.7. It should take less than 10 minutes to solve.

## Data

All datasets are contained within the `dataset_files` folder, with the subfolders below corresponding to the various sections of this study, as outlined in the paper: 

1. `retrospective`: contains the 15-year retrospective dataset, optimized hyperparameters, and subfolders for the splits used in the retrospective study, along with a file containing all retrospective results
2. `post-2021`: contains the held-out dataset of reactions post-mid-2021, along with the model results reported in the paper
3. `monomer_replacement`: contains the dataset of enumerated reactions used to generate yield predictions for the monomer replacement studies, as well as a list of the candidate libraries chosen for the studies
4. `chemist_survey`: contains binary classification and multi-class classification performance metrics for both the model and chemists surveyed

Note that all reaction data has been one-hot encoded (i.e. no molecular structure information), so not all of the provided code will be able to be run.

## General Notes and Disclaimer

1. Several parts of the code require some combination of the following inputs: `model_type`, `split_type`, `task_type`, `feature_type`. For each, certain keywords can be used:

    * `model_type`: can take as input `'rf'`, `'xgb'`, or `'nn'` (for the Random Forest, Gradient Boosting, and feedforward Neural Network, respectively)
    * `split_type`: can take as input `'random'`, `'monomer'`, or `'core'` (for the random, monomer, and core-based splits, respectively)
    * `task_type`: can take as input `'bin'`, `'mul'`, or `'reg'` (for binary classification, multi-class classification, and regression, respectively)
    * `feature_type`: can take as input `'ohe'`, `'fgp'`, `'dft'`, or `'fgpdft'` (for one-hot encoded, fingerprint-only, DFT-only, and concatenated fingerprint and DFT features, respectively)

    Note that because the datasets have been one-hot encoded, non-Abbvie personnel will only be able to run much of the code using `'ohe'` for the `feature_type` field.

2. The _espsim_ package was used to calculate similarity scores in `src/utils/monomer_similarity.py`. This environment required to use this package has certain version conflicts with packages used for our modeling efforts, so to run the similarity calculations, we installed a fresh environment from the _espsim_ Github page (https://github.com/hesther/espsim). Please follow the instructions in their README.md file to setup the environment and replicate our similarity score calculations.

3. This code was tested on a few different machines, and we observed slight numerical variations across different hardware architectures.

## Code Organization

The code is contained within the `src` folder. A description of each subfolder is given here:

### autoqchem_boosted

This is an automated pipeline for generating molecule and atom-level DFT features. It is built on top of _autoqchem_ from the Doyle Lab (see https://github.com/doyle-lab-ucla/auto-qchem), - much of the code is theirs. However, our implementation as given here adds extra checks for force field convergence and conformer validity, as well as choosing the lowest-energy conformers using xTB. See the paper Supplementary Implementation for more details.

### data

This subfolder contains methods for reading (`read_data.py`), splitting (`split.py`), and featurizing (`featurize.py`) the data. Specifically, `split.py` was used to generate the splits given in `dataset_files/retrospective/splits`; `featurize.py` is used to generate one-hot-encoded and fingerprint features, as well as extract the relevant molecule-level and atom-level DFT features, assuming they have already been generated for all reactions. See `autoqchem_boosted` and `utils/extract_dft_descriptors.py` for more details on DFT descriptor generation and handling.

### models

This subfolder contains implementations of the models described in the paper and SI, as well as the code used to run the model-based experiments. 

1. `ffnn.py` contains the feedforward neural network implementation in PyTorch.
2. `main_models.py` is a wrapper class that generating features for, and training and testing, all 3 model types (RF, xGB, NN) given train/val/test data. It uses the respective models given from scikit-learn, xgboost, and `ffnn.py`, and the featurization methods implemented in `featurize.py`.
3. `hyperparam_opt.py` performs hyperparameter optimization to generate the hyperparameters given in `dataset_files/retrospective/optimal_hyperparameters.csv`, which were used for the retrospective and post-2021 data modeling. Please see the SI for a detailed description of the hyperparameter optimization workflow.
4. `gnn_cgr.py` is used to train and test a Chemprop (https://github.com/chemprop/chemprop) model, as described in the SI. The _chemprop_ package was directly used for this.

All other files in this folder simply run the models across various data splits to generate the results shown in the paper. Each file is labeled by the section in the paper it belongs to (i.e. `retrospective_modeling.py` was used to generate the results shown in the "Retrospective Modeling" section of the paper).

### utils

This subfolder contains additional helpful functions for this study:

1. `atom_mapping.py` atom maps a reaction given substrate and product SMILES
2. `extract_dft_descriptors.py` is used to get all molecule-level, and reactive site, atom-level DFT descriptors for a reaction from the completed and extracted Gaussian output files. It then adds the descriptors to the full dataset .csv file. Note that these descriptors are not available for the public release of this repository and paper, to conceal proprietary structures; however, histograms of the descriptors are available in the SI of the paper.
3. `monomer_similarity.py` uses the _espsim_ package (https://github.com/hesther/espsim) to generate a similarity score between 2 molecules on the basis of 3D shape and electrostatic potentials.
4. `plotting.py` generates confusion matrices and parity plots, and saves .pdf versions of figures to be exported into Illustrator.
