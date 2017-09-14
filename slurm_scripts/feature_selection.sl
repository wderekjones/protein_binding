#!/bin/bash -l

source activate protein_binding

python feature_selection.py --out "debug_output/full_set_feature_selection/" --f "data/all_kinase/with_pocket/full_kinase_set_features_list.csv" --data "data/all_kinase/with_pocket/full_kinase_set.h5" --null "data/all_kinase/with_pocket/null_column_list.csv" --strat "mean"

