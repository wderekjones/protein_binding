import pandas as pd
import argparse


parser = argparse.ArgumentParser(description = "Process files containing protein binding affinity features")
parser.add_argument('--f',type = str,nargs='+',help="list paths of files containing protein-drug compound features")
parser.add_argument('--m',type=str, help = "file containing molecular descriptors")
parser.add_argument('--feats',type = str, help ="file containing which molecular descriptors to keep")   # need to add default behavior, i.e. if no file is passed then use all features
args = parser.parse_args()


def read_input_files():
    #create an empty list to store the dataframes
    df_list = []

    #for each input file path, load the dataframe then append to the dataframe list
    for path in args.f:
        df = parse_file(path)
        df_list.append(df)

    #do a pairwise merge (inner join) for each dataframe in the dataframe list
    df_agg = reduce(lambda x,y:pd.merge(x,y,on=["moleculeName","proteinName"]),df_list)

    mol_df = parse_file(args.m)

    output_df = pd.merge(df_agg,mol_df, on="moleculeName")

    #output the aggregated dataframe to .csv
    #TODO: fix the formatting of the output file to have the keys justified to the left and data following on the right
    output_df.to_csv('ml_features.csv',sep=' ')

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
        data = load_molecular_descriptors(filepath,args.feats)
    elif filepath.find('protein_features_coach_avg') != -1: # not sure
        data =  load_protein_features_coach_avg(filepath)
    elif filepath.find('protein_features_2struc') != -1: # not sure
        data =  load_protein_features_2struc(filepath)
    else:
        print 'File not supported'

    return data

# create a function for reading each input file type

def load_molecular_descriptors(filepath,descriptorsListFile):
    '''
        reads input files containing molecular descriptors
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    
    # Fatemah Code
    data = pd.read_csv(filepath,delimiter='\t', low_memory=False)

    # rename duplicated moleculars
    for index, row in data.iterrows():
        molName = data.get_value(index,'NAME',takeable=False)
        duplicatedMol = data.loc[data['NAME'] == molName]
        rowNum = len(duplicatedMol.index)
        
        # there are duplicated Mol.
        # rename from second molecule _#
        molIndex = 0
        if (rowNum>1):
            for innerIndex, innerRow in duplicatedMol.iterrows():
                molIndex = molIndex + 1
                if (molIndex>1):
                    newMolName = molName + '_' + str(molIndex)
                    data.set_value(innerIndex,'NAME',newMolName)
    
    # select descriptorsList
    with open(descriptorsListFile) as f:
        descriptorsList = f.read().splitlines()
        
    descriptorsList.append('NAME')
    for index, column in data.iteritems():
        #index =  column name
        if (index not in descriptorsList):
            data.drop(index,axis=1 , inplace=True)
    
    # rename the second column to use it as key in the merge 
    #descriptorsResults.rename(columns={'NAME':'moleculeName'}, inplace = True)
    data.rename(columns={'NAME':'moleculeName'}, inplace = True)

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
    
    # remove .pdbqt from the Ligand 
    data['Ligand'] = pd.DataFrame(data.Ligand.str.replace('.pdbqt',''))
    
    # extract protein name from Ligand and store it in new column 
    data['proteinName'] = data['Ligand'].str.extract('(...._cluster\d+)', expand=True)
    
    # extract molecule name from Ligand and store it in new column
    data['moleculeName'] = data['Ligand'].str.extract('((?<=cluster\d_)\w+)', expand=True)
    
    # rename Energy col
    data.rename(columns={'Energy':'mmgbsaEnergy'}, inplace = True)
    
    data = data.drop('Ligand',1)

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
    
    # remove .pdbqt from the Ligand 
    data['Ligand'] = pd.DataFrame(data.Ligand.str.replace('.pdbqt',''))
    
    # extract protein name from Ligand and store it in new column 
    data['proteinName'] = data['Ligand'].str.extract('(...._cluster\d+)', expand=True)
    
    # extract molecule name from Ligand and store it in new column
    data['moleculeName'] = data['Ligand'].str.extract('((?<=cluster\d_)\w+)', expand=True)
    
     # rename Energy col
    data.rename(columns={'Energy':'dockingEnergy'}, inplace = True)
    
    # do we need to drop Ligand column ??
    data = data.drop('Ligand',1)

    return data

def load_protein_features_2struc(filepath):
    '''
        reads protein_features_2struc
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    data = pd.read_csv(filepath,delimiter=',')
    
    # rename the first column to use it as key in the merge 
    data.rename(columns={'Cluster_Name':'proteinName'}, inplace = True)
    print data.index.name

    return data
    
def load_protein_features_coach_avg(filepath):
    '''
        reads protein_features_coach_avg
    :param filepath: path to the input file
    :return: pandas DataFrame
    '''
    data = pd.read_csv(filepath,delimiter=',')
    
    # rename the first column to use it as key in the merge 
    data.rename(columns={'cluster_name':'proteinName'}, inplace = True)
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

read_input_files()





