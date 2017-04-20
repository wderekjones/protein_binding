import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import *


X = make_data("data/ml_pro_features.csv")
y = make_data("data/ml_pro_labels.csv")

print (X)
print (y)

rforest = RandomForestClassifier(n_estimators=10)

