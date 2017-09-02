#!/home/wdjo224/miniconda3/envs/protein_binding/bin python


import h5py
import pandas as pd
import numpy as np
import glob
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Combine fragmented csv files, output h5")
parser.add_argument('--i',type=str,help="input file paths to csv files")
parser.add_argument('--o', type=str, help="output file name")
args = parser.parse_args()



def save_to_hdf5(data_frame, output_name):

    output_file = h5py.File(output_name+".h5", "w", libver='latest')
    column_names = list(data_frame.columns.values)
    group_names = list(set(data_frame['receptor']))



    for group_name in tqdm(group_names):
        grp = output_file.create_group(str(group_name))
        grp_data = data_frame[data_frame['receptor'] == group_name]
        for feature in iter(column_names):
            if feature is "label":
                grp.require_dataset(str(feature), [grp_data.shape[0], 1], data=np.asarray(grp_data[feature]),
                                        dtype=np.int32)
            elif feature in ['receptor','drugID']:
                grp.require_dataset(str(feature), [grp_data.shape[0], 1],
                                        data=np.asarray(grp_data[feature]),
                                        dtype=h5py.special_dtype(vlen=str))
            else:
                grp.require_dataset(str(feature), [grp_data.shape[0], 1],
                                        data=np.asarray(grp_data[feature]),
                                        dtype=np.float32)


    output_file.close()



df = pd.read_csv(args.i, na_values=['na'], engine='c', header=0,dtype={'amide': np.float32,
       'basic': np.float32,
       'hBondDonor': np.float32,
       'posCharge': np.float32,
       'vsAnion': np.float32,
       'vsCation': np.float32})

save_to_hdf5(df, str(args.o))
