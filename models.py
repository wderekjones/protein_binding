from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing.imputation import Imputer
from utils import *

# need to make sure that each class is included
X_n, y_n = get_negative_data("data/ml_pro_features_labels.csv",3000)
X_p, y_p = get_positive_data("data/ml_pro_features_labels.csv",3000)
X = combine_positive_negative_data(X_n,X_p)
y = combine_positive_negative_data(y_n,y_p)
imputer = Imputer()
X = imputer.fit_transform(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

rforest = RandomForestClassifier(n_estimators=10,class_weight='balanced',oob_score=True)
rforest.fit(X_train, y_train)
forest_preds = rforest.predict(X_test)
forest_confusion = confusion_matrix(y_test,forest_preds,labels = [0,1])
plot_confusion_matrix(forest_confusion,classes=[0,1])
plt.savefig('results/forest_test.png')
forest_accuracy = accuracy_score(y_test,forest_preds)
print ("Random Forest (n=10) accuracy: ",forest_accuracy)

svm = SVC(class_weight='balanced')
svm.fit(X_train,y_train)
svm_preds = svm.predict(X_test)
svm_confusion = confusion_matrix(y_test,svm_preds)
plt.clf()
plot_confusion_matrix(svm_confusion,classes=[0,1])
plt.savefig('results/svm_test.png')
svm_accuracy = accuracy_score(y_test,svm_preds)
print ("SVM accuracy: ",svm_accuracy)

logistic = LogisticRegression(class_weight='balanced')
logistic.fit(X_train,y_train)
logistic_preds = logistic.predict(X_test)
logistic_confusion = confusion_matrix(y_test,logistic_preds)
plt.clf()
plot_confusion_matrix(logistic_confusion,classes=[0,1])
plt.savefig('results/logistic_test.png')
logistic_accuracy = accuracy_score(y_test,logistic_preds)
print ("Logistic Regression accuracy: ",logistic_accuracy)