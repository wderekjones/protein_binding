#!/bin/bash -l

source activate /home/wdjo224/miniconda3/envs/protein_binding/bin

export OMP_NUM_THREADS=4

python --f "AllKinases/full_features_list.csv" --data "data/all_kinase/with_pocket/full_data_with_pocket_float32.h5"

