import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score,precision_score,recall_score,classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import *


def generate_report(report_title,clf, X_test, y_test):
    '''
        Takes a classifier object and evaluates model performance, generates figures and a table of results
    :param clf: 
    :param X: 
    :param y: 
    :return: 
    '''
    preds = clf.predict(X_test)

    output_file = open("results/"+report_title+".txt","w")

    output_file.write(report_title+ "\n")

    output_file.write(classification_report(y_test,preds,target_names=["Class 0", "Class 1"], digits = 4))

    output_file.write("\n"+ clf.__str__())
    output_file.close()

    confusion = confusion_matrix(y_test, preds)
    plt.clf()
    plot_confusion_matrix(confusion, classes=[0, 1], title=report_title)
    plt.savefig("results/" + report_title + ".png")





X_p, y_p = load_data("data/ml_pro_features_labels.csv", 3000, mode=1)

X_n, y_n = load_data("data/ml_pro_features_labels.csv", 3000, mode=0)
X = combine_positive_negative_data(X_n, X_p)
y = combine_positive_negative_data(y_n, y_p)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


rforest = RandomForestClassifier(n_estimators=10)
rforest.fit(X_train,y_train)


svm = SVC()
svm.fit(X_train,y_train)

logistic = LogisticRegression()
logistic.fit(X_train, y_train)



generate_report("Random Forest",rforest,X_test,y_test)
generate_report("SVM",svm,X_test,y_test)
generate_report("Logistic",logistic,X_test,y_test)