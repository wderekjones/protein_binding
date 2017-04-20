# protein_binding

This repository contains a collection of python scripts for computing a drug-protein binding classification.

## Parser
A script that collects various features from multiple input files and outputs a file of protein, drug, protein-drug pair features as specified by the user. 


In order to run the parser:

> python parser.py --p paths/to/protein_features --m /path/to/molec_descriptors --pm paths/to/protein/molec/features --feats /path/to/molec_features_list

where:

       --p specifies the (list) of protein feature filepaths
       --m denotes molecular descriptors file
       --pm specifies the list of protein-molec feature filepaths
       --feats specifies the file containing the features to keep from the E-Dragon molecular descriptors output

example:
> parser.py --p "SampleInputs/protein/protein_features_2struc.csv" "SampleInputs/protein/protein_features_coach_avg.csv" --pm "SampleInputs/1QCF_ML_Features/docking_summary_final.csv" --m "SampleInputs/MolecularDescriptors-Dragon7/outputExclusionMolecularDescriptors.txt" --feats "SampleInputs/MolecularDescriptors-Dragon7/descriptorsListTS2.txt"
	
Note: this script has been verified to run with python 3.6.1



