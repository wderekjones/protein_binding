import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="computes descriptive values of the data")
parser.add_argument("--i",type=str, help="input path to data")
parser.add_argument("--o",type=str,help="output path for results")
args = parser.parse_args()

# read the data
df = pd.read_csv(args.i)

# compute description (not including NaN values by pandas default), write the description to a file
df.describe().to_csv(args.o+"_description.csv")

# find columns with null values, write these columns to a file
pd.DataFrame(df.columns[df.isnull().any(0)]).to_csv(args.o+"_features_with_nulls.csv")

# get counts of null values for each feature
null_counts = df.isnull().sum()
pd.DataFrame(df.isnull().sum()).to_csv(args.o+"_feature_null_counts.csv")


