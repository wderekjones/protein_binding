import pandas as pd

import numpy as np

import os

'''
    Loading the files in order to determine best utilities. Future improvements will aggregate descriptors for a particular ID and
    output a file ready for training machine learning models.
'''
molec_descriptors = pd.read_csv("SampleInputs/MolecularDescriptors/E-DragonOutput7MolecularDescriptors",delimiter='\t',skiprows=2)

data1 = pd.read_csv("SampleInputs/1QCF_cluster1_mmpbsa_energy_avg_5",delimiter='\t')

data2 = pd.read_csv("SampleInputs/1QCF_docking_results",delimiter='\t')



print data2