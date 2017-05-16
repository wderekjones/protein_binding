import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import *

random_state = 0

X_p, y_p = load_data("data/ml_pro_features_labels.csv", 4000, mode=1)

X_n, y_n = load_data("data/ml_pro_features_labels.csv", 100000, mode=0)
X = combine_positive_negative_data(X_n, X_p)
y = combine_positive_negative_data(y_n, y_p)

# y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

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
