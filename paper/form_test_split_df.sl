#!/bin/sh

source activate protein_binding

python form_split_df.py -i /scratch/wdjo224/data/data/kinase/with_pocket/full_kinase_set.csv -split /scratch/wdjo224/protein_binding/paper/x_test_compounds.csv -o /scratch/wdjo224/data/data/kinase/with_pocket/intermediate_test.csv
