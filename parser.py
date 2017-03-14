import pandas as pd
import argparse


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
    data.drop('Order',axis=1,inplace=True)
    return data

def load_docking_results(filepath):
    data = pd.read_csv(filepath,delimiter='\t')
    data.drop('Order',axis=1,inplace=True)
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
            feature_list[col_name] = value
            j+=1

    df = pd.DataFrame([feature_list])
    return df

def get_merged_results(dfX,dfY,key):
    '''

    Wraps the pandas call to merge.

    :param dfX: one of the two dataframes to be merged
    :param dfY: the other of the two dataframes to be merged
    :param key: a string denoting the column on which the dataframes will be merged
    :return: a pandas dataframe containing the matches between the two dataframes (maximum = least row dimenison of dfX or dfY)
    '''
    merge_result = dfX.merge(dfY,left_on=key, right_on=key)
    merge_result = merge_result.set_index(key).reset_index()
    return merge_result


#TODO: write a function to merge the molecular descriptors to the docking and energy results. Will need regular expressions.
#TODO: read the filenames from stdin and then execute necessary functions to generate output datafile

 parser = argparse.ArgumentParser(description='Process some integers.')

## Load the data for each command line argument filepath

# load the e-dragon features
data0 = load_molecular_descriptors(path0)

# load the mmgbsa energy features
data1 = load_mmgbsa_energy(path1)

# load the docking results
data2 = load_docking_results(path2)

merged_data = get_merged_results(data1,data2,'Ligand')

for mol_id in data0['MOL_ID']:
    for ligand_id in data1['Ligand']:
        print ligand_id.find(mol_id)
