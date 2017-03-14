# protein_binding
A script that collects various features from multiple input files and outputs a file of protein, drug, protein-drug pair features as specified by the user. 


In order to run the parser:

ex:	> python parser.py SampleInputs/1QCF_cluster1_mmpbsa_energy_avg_5 SampleInputs/1QCF_docking_results


The parser will take in a number of input strings, determin the file type, and then run the appropriate parsing utility to extract the data from each input file path. 

TODO:
	add functionality for user input arguments that specify whether to generate output file of protein features, drug features, or protein-drug binding features. 

	add merging functionality for drug || protein features to drug && protein compound features

	




