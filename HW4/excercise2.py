import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

n_rows = 300000
df = pd.read_csv("train.csv", nrows=n_rows)
Y = df['click'].values
X = df.drop(['click', 'id', 'hour', 'device_id', 'device_ip'], axis=1).values

n_train = int(n_rows * 0.9)
X_train = X[:n_train]
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

enc = OneHotEncoder(handle_unknown='ignore')
X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)
parameters = {'max_depth': [3, 10, None], 'n_estimators': [100, 500, 10], 'min_samples_split': [30, 50, 80],
              'max_features': ['auto', 10, 15]}  # best params : None, 500, 80, 15

parameters1 = {'n_estimators': [1000, 500, 2000], 'min_samples_split': [120, 100, 80],
               'max_features': ['auto', 15, 19]}  # best params : 500, 120, 19

bestParameters = {'n_estimators': [500], 'min_samples_split': [120],
                  'max_features': [19]}  # best params : 500, 120, 19

random_forest = RandomForestClassifier(criterion='gini', n_jobs=-1)

grid_search = GridSearchCV(random_forest, bestParameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train_enc, Y_train)
print(grid_search.best_params_)
random_forest_best = grid_search.best_estimator_
pos_prob = random_forest_best.predict_proba(X_test_enc)[:, 1]
print('The ROC AUC on testing set is: {0:.3f}'.format(roc_auc_score(Y_test, pos_prob)))
