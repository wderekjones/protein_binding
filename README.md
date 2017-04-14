# protein_binding
A script that collects various features from multiple input files and outputs a file of protein, drug, protein-drug pair features as specified by the user. 


In order to run the parser:

> python parser.py --p paths/to/protein_features -m /path/to/molec_descriptors --pm paths/to/protein/molec/features --feats /path/to/molec_features_list

where:

       --f specifies the (list) of summary input file paths to read
       --feats specifies the file containing the features to keep from the E-Dragon molecular descriptors output
       --m denotes molecular descriptors file

example:
> python parser.py --p "SampleInputs/protein/protein_features_2struc.csv" "SampleInputs/protein/protein_features_coach_avg.csv" --pm "SampleInputs/1QCF_ML_Features/docking_summary_final.csv" --m "SampleInputs/MolecularDescriptors-Dragon7/outputExclusionMolecularDescriptors.txt" --feats "SampleInputs/MolecularDescriptors-Dragon7/descriptorsListTS2.txt"
	




