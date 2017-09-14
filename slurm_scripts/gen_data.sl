#!/bin/bash -l

#SBATCH --mail-user wdjo224@g.uky.edu

source activate protein_binding



python full_parser.py --p "/scratch/wdjo224/protein_binding/AllKinases/drug_miner/drugMiner_g2_g3.csv" "/scratch/wdjo224/protein_binding/AllKinases/drug_miner/expasy.csv" "/scratch/wdjo224/protein_binding/AllKinases/drug_miner/porter.csv" --bp "/scratch/wdjo224/protein_binding/AllKinases/prank/binding_site_feature_vectors.csv" --pm "/scratch/wdjo224/protein_binding/AllKinases/docking/binding_features.csv" --m "/scratch/wdjo224/protein_binding/AllKinases/dragon/dragon7_output.csv" --out_dir "/scratch/wdjo224/protein_binding/data/all_kinase/with_pocket" --o full_kinase_set


python create_h5.py --i data/all_kinase/with_pocket/full_kinase_set.csv --o data/all_kinase/with_pocket/full_kinase_set

zip /scratch/wdjo224/protein_binding/data/all_kinase/with_pocket/full_kinase_set.zip /scratch/wdjo224/protein_binding/data/all_kinase/with_pocket/full_kinase_set.h5


