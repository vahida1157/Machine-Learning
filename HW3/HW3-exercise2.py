from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

face_data = fetch_lfw_people(min_faces_per_person=50)

X = face_data.data
Y = face_data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = SVC(class_weight='balanced', random_state=42)
parameters3 = {'C': [9], 'gamma': [1e-07], 'kernel': ['rbf']}  # C : 9 -> 88.1%

grid_search = GridSearchCV(clf, parameters3, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)

print('The best model:\n', grid_search.best_params_)

clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)

print(f'The accuracy is: {clf_best.score(X_test, Y_test) * 100:.1f}%')  # 83.1
