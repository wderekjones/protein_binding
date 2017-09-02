### protein_binding: developer log
#### Derek Jones
06/26/17
* decided that index formatting should be taken care of prior to becoming input to the 
parser as it can become expensive to reset indices, rename columns, and etc;.
* attempted to redo the code using dask dataframes. Did not see a substantial speedup 
when parsing the dragon input so decided to continue using pandas
* experimented with numba. Also did not see mcuh of a speedup due to the fact that
much of pandas is implemented in cython already.

06/28/17
* seem to be making progress on finishing the new parser. Moved much of the preprocessing
to the functions that I have written to aggregate the data. This way the parser 
can expect roughly the same form of input each time. Doing this also increases generality, so as long as the input files have some subset of the 
key values (should supply these as arguments rather than hardcoding), then any input can be used.
* still need to add the labels, rename the features from group 2/3 of drugminer, and the additional features from group 1 of the drug miner paper

06/29/30
* today I changed the structure of the parser code so that the arg checks and
merging happen in a much more logical manner than before. Now, protein-drug
features are read in first, followed by a check for protein/drug features (which are merged if there any 
to be merged). Assuming that all entries in the protein-drug design matrix are unique, then the number of
entries should remain constant as there are 26 unique protein entries that are mapped to the 
~300000 unique protein-drug entries. Then the unique entries from the dragon output are mapped to the same set. Both operations
require the use of inner-joins. For proteins the inner join is appropriate because each
of the 26 receptors maps to at least 1 element in the protein-drug features set, so there is no reason to use anything else. A similar
case can be made for the drug data. Assuming that all entries are unique and that the size
of the set is < size of the protein_drug set, then using an inner join will yield a result 
that is the same size as the protien-drug matrix. Furthermore, in layman's terms,
each data element in our dataset consists of numeric values corresponding to a specific receptor and a specific drug. Therefore if 
we have each piece of the puzzle for each of the reactions then we should be able to use each of our interactions as an example for machine learning models. 
* In other news, I removed the functions for loading each specific type of features (protein,drug,protein-drug) and 
replaced with one function that loads each in the same manner. This is possible due to how the code is structured and the
addition of the preprocessing scripts that will be used to prepare each type of data for the parser.
* decided to remove all inplace=True operations in order to decrease running time. memory
usage is becoming less of a concern when running time has scaled proportionally
 with increased input size.
* Goal: finish changes and generate visualizations by 06/30/17?
* Goal: beginning next week, visualize and learn on the new dataset.