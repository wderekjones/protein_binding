import argparse
import pandas as pd
import numpy as np
from utils.input_pipeline import compute_proportion


parser = argparse.ArgumentParser(description="compute feature proportions")

#parser.add_argument("--F",type=str,help="input file path to full feature set")
parser.add_argument("-f",type=str,help="input file path to feature subset")

args = parser.parse_args()


root_path = "/scratch/wdjo224/data/data/kinase/with_pocket/"

print("feature file: ", args.f)
feature_subset = pd.read_csv(args.f,header=None)

# load the full feature set
full_features = pd.read_csv(root_path+"full_kinase_set_features_list.csv",header=None)
print("proportion of features in full feature set:", compute_proportion(list(feature_subset[0]),list(full_features[0])))

# compute the number of features from the subset that belong to the binding group
binding_features = pd.read_csv(root_path+"binding_features_list.csv", header=None)
print("proportion of features in binding feature set:", compute_proportion(list(feature_subset[0]),list(binding_features[0])))

# compute the number of features from the subset that belong to the drug group
dragon_features = pd.read_csv(root_path+"drug_features_list.csv",header=None)
print("proportion of features in drug feature set:", compute_proportion(list(feature_subset[0]),list(dragon_features[0])))

# compute the number of features from the subset that belong to the protein group
drugminer_features = pd.read_csv(root_path+"protein_features_list.csv",header=None)
print("proportion of features in protein feature set:", compute_proportion(list(feature_subset[0]),list(drugminer_features[0])))

# compute the number of features from the subset that belong to the binding pocket group
prank_features = pd.read_csv(root_path+"pocket_features_list.csv",header=None)
print("proportion of features in pocket feature set:", compute_proportion(list(feature_subset[0]),list(prank_features[0])))