#!/usr/bin/env bash

source activate protein_binding

python convert_csv_to_h5.py -i /scratch/wdjo224/data/data/kinase/with_pocket/full_kinase_set_no_duplicates.csv -o /scratch/wdjo224/data/data/kinase/with_pocket/kinase_no_duplicate.h5 -c kinase_no_duplicate