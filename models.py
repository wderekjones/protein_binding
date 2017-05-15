import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import *

random_state = 0

X, y = load_data("data/ml_pro_features_labels.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

rforest = RandomForestClassifier(n_estimators=10, random_state=random_state)
rforest.fit(X_train, y_train)
rforest_preds = rforest.predict(X_test)

svm = SVC(random_state=random_state)
svm.fit(X_train, y_train)
svm_preds = svm.predict(X_test)

logistic = LogisticRegression(random_state=random_state)
logistic.fit(X_train, y_train)
logistic_preds = logistic.predict(X_test)

generate_report("Random Forest", rforest, rforest_preds, y_test)
generate_report("SVM", svm, svm_preds, y_test)
generate_report("Logistic", logistic, logistic_preds, y_test)
