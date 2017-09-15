#!/bin/bash -l
source activate protein_binding

python feature_selection.py  --out "debug_output/top_selection_forest_feature_selection/" --f "debug_output/union_top_subset_features.csv" --null "data/all_kinase/with_pocket/more_than_5_percent_missing_features_list.csv" --strat "mean" --data "data/all_kinase/with_pocket/full_kinase_set.h5"
