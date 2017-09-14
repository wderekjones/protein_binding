#!/bin/bash -l

source activate protein_binding
python combine_h5.py --i data/all_kinase/with_pocket/full_data_with_pocket --o data/all_kinase/with_pocket/full_data_with_pocket_float32

zip data/all_kinase/with_pocket/full_data_with_pocket_float32.zip data/all_kinase/with_pocket/full_data_with_pocket_float32.h5
