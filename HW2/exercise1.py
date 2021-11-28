import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

decisionTree_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
decisionTree_classifier.fit(X_train, y_train)

knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_classifier.fit(X_train, y_train)

kernel_svm = SVC(kernel='rbf', random_state=0)
kernel_svm.fit(X_train, y_train)

Logistic = LogisticRegression(random_state=0)
Logistic.fit(X_train, y_train)

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

RandomForest_Classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
RandomForest_Classifier.fit(X_train, y_train)

svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(X_train, y_train)

decisionTree_classifier_pred = decisionTree_classifier.predict(X_test)
cm = confusion_matrix(y_test, decisionTree_classifier_pred)
print(cm, "\nDecision Tree: ", accuracy_score(y_test, decisionTree_classifier_pred))

svm_classifier_pred = svm_classifier.predict(X_test)
cm = confusion_matrix(y_test, svm_classifier_pred)
print(cm, "\nSVM: ", accuracy_score(y_test, svm_classifier_pred))

kernel_svm_pred = kernel_svm.predict(X_test)
cm = confusion_matrix(y_test, kernel_svm_pred)
print(cm, "\nKernel SVM: ", accuracy_score(y_test, kernel_svm_pred))

knn_classifier_pred = knn_classifier.predict(X_test)
cm = confusion_matrix(y_test, knn_classifier_pred)
print(cm, "\nK Nearest Neighbor: ", accuracy_score(y_test, knn_classifier_pred))

Logistic_pred = Logistic.predict(X_test)
cm = confusion_matrix(y_test, Logistic_pred)
print(cm, "\nLogistic Regression: ", accuracy_score(y_test, Logistic_pred))

naive_bayes_pred = naive_bayes_classifier.predict(X_test)
cm = confusion_matrix(y_test, naive_bayes_pred)
print(cm, "\nNaive Bayes: ", accuracy_score(y_test, naive_bayes_pred))

RandomForest_pred = RandomForest_Classifier.predict(X_test)
cm = confusion_matrix(y_test, RandomForest_pred)
print(cm, "\nRandom Forest: ", accuracy_score(y_test, RandomForest_pred))
