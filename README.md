# protein_binding
A script that collects various features from multiple input files and outputs a file of protein, drug, protein-drug pair features as specified by the user. 

The parser will take in a number of input strings, determine the file type, and then run the appropriate parsing utility to extract the data from each input file path.

In order to run the parser:

ex:	> python parser.py --f SampleInputs/1QCF_cluster1_mmpbsa_energy_avg_5 SampleInputs/1QCF_docking_results /molculardescriptors --feats /path/to/file/containing/features

where: --f specifies the (list) of input file paths to read
       --feats specifies the file containing the features to keep from the E-Dragon molecular descriptors output\



	




