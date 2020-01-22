import pandas as pd
x_train = pd.read_pickle("preprocessed_data/x_train")
y_train = pd.read_pickle("preprocessed_data/y_train").values.ravel() # transforms the collumns in a np array with shape (nsamples,)
x_test = pd.read_pickle("preprocessed_data/x_test")
y_test = pd.read_pickle("preprocessed_data/y_test").values.ravel()


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
cross_val_scores = cross_val_score(gnb, x_train, y_train)
print("Naive bayes classifier cross validation score:", cross_val_scores)
gnb.fit(x_train, y_train)
predictions = gnb.predict(x_test)
print("accuracy: ", accuracy_score(y_test, predictions))
print("recall: ", recall_score(y_test, predictions))
print("f1 score: ", f1_score(y_test, predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
print("Area under ROC curve: ", roc_auc_score(y_train, cross_val_predict(gnb, x_train, y_train)))




from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
cross_val_scores = cross_val_score(knn_clf, x_train, y_train)
print("\n\nK neighbors classifier cross validation score:", cross_val_scores)
knn_clf.fit(x_train, y_train)
predictions = knn_clf.predict(x_test)
print("accuracy: ", accuracy_score(y_test, predictions))
print("recall: ", recall_score(y_test, predictions))
print("f1 score: ", f1_score(y_test, predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
print("Area under ROC curve: ", roc_auc_score(y_train, cross_val_predict(knn_clf, x_train, y_train)))


from sklearn.tree import DecisionTreeClassifier
dt_cf = DecisionTreeClassifier()
cross_val_scores = cross_val_score(dt_cf, x_train, y_train)
print("\n\nDecision tree cross validation score:", cross_val_scores)
dt_cf.fit(x_train, y_train)
predictions = dt_cf.predict(x_test)
print("accuracy: ", accuracy_score(y_test, predictions))
print("recall: ", recall_score(y_test, predictions))
print("f1 score: ", f1_score(y_test, predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
print("Area under ROC curve: ", roc_auc_score(y_train, cross_val_predict(dt_cf, x_train, y_train)))

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
cross_val_scores = cross_val_score(rf_clf, x_train, y_train)
print("\n\nRandom forest classifier cross validation score:", cross_val_scores)
rf_clf.fit(x_train, y_train)
predictions = rf_clf.predict(x_test)
print("accuracy: ", accuracy_score(y_test, predictions))
print("recall: ", recall_score(y_test, predictions))
print("f1 score: ", f1_score(y_test, predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, predictions))
print("Area under ROC curve: ", roc_auc_score(y_train, cross_val_predict(rf_clf, x_train, y_train)))
