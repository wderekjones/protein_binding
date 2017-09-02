#!/bin/bash -l

source activate protein_binding

python /scratch/wdjo224/protein_binding/gen_tsne_visualization.py --f "/scratch/wdjo224/protein_binding/data/all_kinase/with_pocket/protein_features_list.csv" --i "/scratch/wdjo224/protein_binding/data/all_kinase/with_pocket/full_kinase_set.h5"