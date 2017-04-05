# protein_binding
A script that collects various features from multiple input files and outputs a file of protein, drug, protein-drug pair features as specified by the user. 

The parser will take in a number of input strings, determine the file type, and then run the appropriate parsing utility to extract the data from each input file path.

In order to run the parser:

> python parser.py --f /path/to/docking_energy /path/to/mmpbsa_energy -m /path/to/molec_descriptors --feats /path/to/molec_features

where:

       --f specifies the (list) of input file paths to read
       --feats specifies the file containing the features to keep from the E-Dragon molecular descriptors output
       --m denotes molecular descriptors file

example:
> python parser.py --m "SampleInputs/MolecularDescriptors-Dragon7/outputAllDescriptors.txt" --f "SampleInputs/1QCF_cluster1_mmpbsa_energy_avg_5" "SampleInputs/1QCF_docking_results" --feats "SampleInputs/MolecularDescriptors-Dragon7/outputExclusionDescriptors.txt"

	




