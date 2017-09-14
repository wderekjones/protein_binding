#!/bin/bash -l
source activate protein_binding

python feature_selection.py  --out "debug_output/drug_feature_selection/" --f "data/all_kinase/with_pocket/drug_features_list.csv" --null "data/all_kinase/with_pocket/null_column_list.csv" --strat "mean" --data "data/all_kinase/with_pocket/full_kinase_set.h5"
