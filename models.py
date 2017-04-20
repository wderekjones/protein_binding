import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils import *

# need to make sure that each class is included
X, y = make_data("data/ml_pro_features_labels.csv",100)


X_train,X_test,y_train,y_test = train_test_split(X,y)

rforest = RandomForestClassifier(n_estimators=10)

rforest.fit(X_train, y_train)

preds = rforest.predict(X_test)

confusion = confusion_matrix(y_test,preds)

plot_confusion_matrix(confusion)

plt.show()
