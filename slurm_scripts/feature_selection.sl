#!/bin/bash -l
#SBATCH -J feature_selection
#SBATCH -o slurm_output/feature_selection_%A_%.out

source activate protein_binding

python feature_selection.py --out "debug_output/full_set_feature_selection/"  --f "data/all_kinase/with_pocket/full_kinase_set_features_list.csv" --data "data/all_kinase/with_pocket/full_kinase_set.h5" --null "data/all_kinase/with_pocket/more_than_5_percent_missing_features_list.csv" --strat "mean"
