import pandas as pd
import argparse
import sys


def read_input_files():
    df_dict = {}
    file_index = 0

    for path in sys.argv[1:]:
        # need to merge molecular descriptors...
        if file_index > 0:
            df = parse_file(path)
            df_dict[file_index] = df
        file_index +=1

    df_agg = pd.concat(df_dict,axis=0)
    df_agg.shape
    df_agg.drop_duplicates(inplace=True)

    # output the aggregated file, need to find a better name...
    df_agg.to_csv('ml_features.csv',sep=' ')

    # TODO: verfy that there are no duplicates




def parse_file(filepath):
    '''
        Given a filepath, determines which parser to run in order to extract data. Returns a pandas DataFrame.
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    data = pd.DataFrame()
    if filepath.find('.dat') != -1:
        print filepath.find('.dat')
        data = load_dat(filepath)
    elif filepath.find('.pdbqt') != -1:
        print 'Not implemented'
    elif filepath.find('mmpbsa_energy') != -1:
        data = load_mmgbsa_energy(filepath)
    elif filepath.find('docking_results') != -1:
        data = load_docking_results(filepath)
    elif filepath.find('MolecularDescriptors') != -1:
        data = load_molecular_descriptors(filepath)
    else:
        print 'File not supported'

    return data

# create a function for reading each input file type

def load_molecular_descriptors(filepath):
    '''
        reads input files containing molecular descriptors
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    data = pd.read_csv(filepath,delimiter='\t',skiprows=2)
    data = data.set_index('MOL_ID').reset_index()
    return data

def load_mmgbsa_energy(filepath):
    '''
        reads output from mmgbsa calculations
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    data = pd.read_csv(filepath,delimiter='\t')
    data.drop('Order',axis=1,inplace=True)
    data = data.set_index('Ligand').reset_index()
    return data

def load_docking_results(filepath):
    '''
        reads docking results
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    data = pd.read_csv(filepath,delimiter='\t')
    data.drop('Order',axis=1,inplace=True)
    data = data.set_index('Ligand').reset_index()
    return data

def load_dat(filepath):
    '''
        Loads VINA results

    :param filepath: path to file containing data
    :return: a pandas dataframe of VINA results
    '''
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

    Merge two dataframes (datadrames assumed to have common key, does not support protein || drug and protein && drug matching).

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
#TODO: merge all resulting dataframes on id's

def merge_molec_descriptors(dfM,dfX):
    # for each m_index in dfM
        # for each x_index in dfX
            #if m_index in x_index
                #concatenate dfM[m_index] to dfX[x_index]

    #for m_index in dfM['MOL_ID'].values:
    for m_row in dfM.itertuples():
        #print m_row
    #    for x_index in dfX['Ligand'].values:
        for x_row in dfX.itertuples():
            print x_row[1]
            #if m_index in x_index:
            #    print type(pd.concat([dfM[m_index],dfX[x_index]],axis=1))

    return

read_input_files()





