from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing.imputation import Imputer
from utils import *

# need to make sure that each class is included
X_n, y_n = get_negative_data("data/ml_pro_features_labels.csv",200)
X_p, y_p = get_positive_data("data/ml_pro_features_labels.csv",200)


print (y_p)
print (y_n)

X = combine_positive_negative_data(X_n,X_p)
y = combine_positive_negative_data(y_n,y_p)

imputer = Imputer()
X = imputer.fit_transform(X,y)
X_train,X_test,y_train,y_test = train_test_split(X,y)
rforest = RandomForestClassifier(n_estimators=10,class_weight='balanced',oob_score=True)
rforest.fit(X_train, y_train)
preds = rforest.predict(X_test)
confusion = confusion_matrix(y_test,preds,labels = [0,1])
plot_confusion_matrix(confusion,classes=[0,1])
plt.savefig('test.png')

accuracy = accuracy_score(y_test,preds)

print ("Accuracy: ",accuracy)