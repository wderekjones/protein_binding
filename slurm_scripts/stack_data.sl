#!/bin/bash -l

source activate /home/wdjo224/miniconda3/envs/protein_binding/bin

python combine_h5.py --i data/all_kinase/with_pocket/full_data_with_pocket --o data/all_kinase/with_pocket/full_data_with_pocket_float32

zip data/all_kinase/with_pocket/full_data_with_pocket_float32.zip data/all_kinase/with_pocket/full_data_with_pocket_float32.h5
