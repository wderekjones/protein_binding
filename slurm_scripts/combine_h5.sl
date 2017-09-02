#!/bin/bash -l

#SBATCH -p Short
#SBATCH --nodes=1

export OMP_NUM_THREADS=2

python combine_h5.py --i /scratch/wdjo224/protein_binding/data/all_kinase/with_pocket --o /scratch/wdjo224/protein_binding/data/all_kinase/with_pocket