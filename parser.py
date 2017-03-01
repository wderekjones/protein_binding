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

path3 = "SampleInputs/1QCF_cluster1_CHEMBL568101.1.dat"


# create a function for reading each input file type

def load_molecular_descriptors(filepath):

	data = pd.read_csv(filepath,delimiter='\t',skiprows=2)

	return data


def load_mmgbsa_energy(filepath):
	data = pd.read_csv(filepath,delimiter='\t')

	return data

def load_docking_results(filepath):
	data = pd.read_csv(filepath,delimiter='\t')
	return data


def load_dat(filepath):
    file = open(filepath)

    feature_list = {}

    i = 0
    j = 0

    for line in file.readlines():
        if line.find("TOTAL") != -1 and line.find("DELTA") == -1:
            line = line.strip('\n')
            line = line.split(" ")
            line = filter(lambda x: len(x) >0,line)
            #print line
            value = float(line[1])
            line[0] = line[0]+str(i)
            col_name = line[0]
            feature_list[col_name] = value
            i +=1
        elif line.find("TOTAL") != -1 and line.find("DELTA") != -1:
            line = line.strip('\n')
            line = line.split(" ")
            line = filter(lambda x: len(x) > 0, line)
            line[1] = line[1]+str(j)
            col_name = line[0]+line[1]
            value = float(line[2])
            f_list = (col_name,value)

            feature_list[col_name] = value
            j+=1


    df = pd.DataFrame([feature_list])
    return df

'''

	take as input a number of data frames and an identifier. Create a new dataframe from each input dataframe (list of dataframes?) which contains each column for the 
	identifier. return as numpy array.

def merge_results_on_id():
	

'''

# test the functions

# load the e-dragon features
data0 = load_molecular_descriptors(path0)
print data0

data1 = load_mmgbsa_energy(path1)
print data1


data2 = load_docking_results(path2)
print data2

data3 = load_dat(path3)
print data3


