#!/usr/bin/env bash

source activate protein_binding

python convert_kinase_h5.py -i data/all_kinase/with_pocket/full_kinase_set.h5 -o data/all_kinase/with_pocket/full_kinase_set_80_20_split.h5 -train "x_train_compounds.csv" -test "x_test_compounds.csv"
