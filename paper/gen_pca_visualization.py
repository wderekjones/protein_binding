import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
plt.style.use("seaborn-muted")
from utils.input_pipeline import load_data
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA


parser = argparse.ArgumentParser()
parser.add_argument("--i",type=str,help="input file path to the data")
parser.add_argument("--f",type=str,help="input file path to the feature set")


args = parser.parse_args()

feature_set = pd.read_csv(args.f)
# hacked solution
feature_set = feature_set['0']

X,y = load_data(args.i,features_list=feature_set)

# preprocess the data using PCA, reduce by an order of magnitude
X_pca = IncrementalPCA(n_components=int(np.floor(X.shape[1]/10))).fit_transform(X)

# Visualize the results

plt.scatter(X_pca[:,0],X_pca[:,1],c = y.flatten())

plt.savefig("test_pca.png")