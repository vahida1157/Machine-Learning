import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('heart.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

gaussianNB = GaussianNB()
gaussianNB.fit(X_train, y_train)

gaussianNb_predict = gaussianNB.predict(X_test)
cm = confusion_matrix(y_test, gaussianNb_predict)
print(cm, "\nNaive Bayes(GaussianNB): ", accuracy_score(y_test, gaussianNb_predict))

bernoulliNB = BernoulliNB()
bernoulliNB.fit(X_train, y_train)

bernoulliNB_predict = bernoulliNB.predict(X_test)
cm = confusion_matrix(y_test, bernoulliNB_predict)
print(cm, "\nNaive Bayes(BernoulliNB): ", accuracy_score(y_test, bernoulliNB_predict))
