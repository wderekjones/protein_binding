#!/usr/bin/env bash

source activate protein_binding

python convert_csv_to_h5.py -i data/all_kinase/with_pocket/full_kinase_set.csv -o data/all_kinase/with_pocket/kinase.h5