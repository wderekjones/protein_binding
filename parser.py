import pandas as pd

import numpy as np

import os

'''
    Loading the files in order to determine best utilities. Future improvements will aggregate descriptors for a particular ID and
    output a file ready for training machine learning models.
'''

path0 = "SampleInputs/MolecularDescriptors/E-DragonOutput7MolecularDescriptors"

path1 = "SampleInputs/1QCF_cluster1_mmpbsa_energy_avg_5"

path2 = "SampleInputs/1QCF_docking_results"



def load_molecular_descriptors(filepath):

	data = pd.read_csv(filepath,delimiter='\t',skiprows=2)

	return data


def load_mmgbsa_energy(filepath):
	data = pd.read_csv(filepath,delimiter='\t')

	return data

def load_docking_results(filepath):
	data = pd.read_csv(filepath,delimiter='\t')
	return data


'''

	take as input a number of data frames and an identifier. Create a new dataframe from each input dataframe (list of dataframes?) which contains each column for the 
	identifier. return as numpy array.

def merge_results_on_id():
	

'''

data0 = load_molecular_descriptors(path0)


data1 = load_mmgbsa_energy(path1)

data2 = load_docking_results(path2)





