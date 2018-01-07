# protein_binding

This repository contains the implementation of "Polypharmacology Within the Full Kinome: a Machine Learning Approach".
## Parser
A script that collects various features from multiple input files and outputs a file of protein, drug, protein-drug pair features as specified by the user. 


In order to run the parser:

> python full_parser.py --p paths/to/protein_features --m /path/to/molec_descriptors --pm paths/to/protein/molec/features --feats /path/to/molec_features_list

where:

       --p specifies the (list) of protein feature filepaths
       --m denotes molecular descriptors file
       --pm specifies the list of protein-molec feature filepaths
       --feats specifies the file containing the features to keep from the E-Dragon molecular descriptors output
       --o specifies the output file name

Note that each of the arguments are optional.

example:
> parser.py --p "SampleInputs/protein/protein_features_2struc.csv" "SampleInputs/protein/protein_features_coach_avg.csv" --pm "SampleInputs/1QCF_ML_Features/docking_summary_final.csv" --m "SampleInputs/MolecularDescriptors-Dragon7/outputExclusionMolecularDescriptors.txt" --feats "SampleInputs/MolecularDescriptors-Dragon7/descriptorsListTS2.txt"
	
Note: this script has been verified to run with python 3.6.1

---
## Feature Selection

To run the random forest feature selection: 

> python feature_selection.py -f -data -null -strat -label -out -prot -names -root

where:

        -f: path(s) to set of initial features
        -data: path to initial dataset, this should at least contain the features specified by --f if given
        -null: path to null features
        -strat: imputation strategy used to fill in the null values
        -label: optional specify target label
        -out: output path to dir
        -prot: a flag that indicates that protein names are used as labels
        -names: list of proteins to exclude from training
        -root: root path for data and feature lists
        --split: if using dataset with seperate train/test splits, specify the split"

