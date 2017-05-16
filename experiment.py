import h5py

from utils import *


def balanced_experiment(data_path, sample_size=None, features_list=None):
    input_fo = h5py.File(data_path, 'r')

    if sample_size is None:
        sample_size = 199752

    if features_list is None:
        features_list = list(input_fo.keys())

    data_array = np.zeros([sample_size,len(features_list),1])

    # get the data and store in numpy array
    #if features_list not None:

    #else
    i = 0
    for dataset in input_fo.keys():
        #print(dataset)
        data = input_fo[dataset]
        data = np.asarray(data)
        if data.shape[0] > 10:
#       print (data.shape)
            data = np.reshape(data,[data.shape[0]])
        #print (data.shape)
            data_array[:,i,0] = data
        i+=1

    return data_array

feat_list = ["cluster_number",
"cluster_number_x",
"cluster_number_y"]

balanced_experiment("data/ml_pro_features_labels.h5")

def sample_from_datasets():
    '''
    TODO: create a function that returns a random sample from the datasets. either a list of feature strings are specified
    or all of the features are used
    :return: 
    '''

    return 0