from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

clf = SVC(class_weight='balanced', random_state=42)
parameters = {'C': [0.1, 1, 10], 'gamma': [1e-07, 5e-07, 9e-07, 1e-08, 1e-06], 'kernel': ['rbf', 'linear']}  # 1e-07
# -> 87.7%
parameters1 = {'C': [0.1, 1, 10], 'gamma': [1e-07, 5e-07, 9e-07], 'kernel': ['rbf', 'linear']}  # 1e-07 -> 87.7%
parameters2 = {'C': [0.1, 1, 10], 'gamma': [1e-07, 5e-06, 9e-06], 'kernel': ['rbf', 'linear']}  # 1e-07 -> 87.7%
parameters3 = {'C': [9, 10, 11], 'gamma': [1e-07], 'kernel': ['rbf']}  # C : 9 -> 88.1%
parameters4 = {'C': [7, 8, 9], 'gamma': [1e-07], 'kernel': ['rbf']}  # C : 8 -> 87.7%

grid_search = GridSearchCV(clf, parameters3, n_jobs=-1, cv=5)
grid_search.fit(X_train, Y_train)

print('The best model:\n', grid_search.best_params_)

clf_best = grid_search.best_estimator_
pred = clf_best.predict(X_test)

print(f'The accuracy is: {clf_best.score(X_test, Y_test) * 100:.1f}%')

# print(classification_report(Y_test, pred, target_names=face_data.target_names))
#
# # True : good. False : error -- n_components : 100 -> 90.5% , 10 -> 46.3
# pca = PCA(n_components=169, whiten=True, random_state=42)
#
# svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)
#
# model = Pipeline([('pca', pca), ('svc', svc)])
#
# parameters_pipeline = {'svc__C': [1, 3, 10],
#                        'svc__gamma': [0.001, 0.005]}  # {'svc__C': 1, 'svc__gamma': 0.005} -> 90.2%
# parameters_pipeline1 = {'svc__C': [1, 2, 3, 10],
#                         'svc__gamma': [0.001, 0.005, 0.006, 0.004]}  # {'svc__C': 3, 'svc__gamma': 0.006} -> 90.5%
# grid_search = GridSearchCV(model, parameters_pipeline1)
# grid_search.fit(X_train, Y_train)
# print(grid_search.best_params_)
# model_best = grid_search.best_estimator_
# print(f'The accuracy is: {model_best.score(X_test, Y_test) * 100:.1f}%')
# pred = model_best.predict(X_test)
# print(classification_report(Y_test, pred, target_names=face_data.target_names))
