#!/bin/bash -l
source activate protein_binding

python feature_selection.py  --out "debug_output/top_20_feature_selection/" --f "data/all_kinase/with_pocket/Informative_features.csv" --null "data/all_kinase/with_pocket/more_than_5_percent_missing_features_list.csv" --strat "mean" --data "data/all_kinase/with_pocket/full_kinase_set.h5"
