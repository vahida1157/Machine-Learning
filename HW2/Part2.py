from sklearn.naive_bayes import BernoulliNB
import numpy as np

# region DATA
# create train sample
X_train = np.array([
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])

# create label for our train sample
Y_train = np.array(['Y', 'N', 'Y', 'Y'])

# create test case model
X_test = np.array([[1, 1, 0]])

# endregion

clf = BernoulliNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)
pred_prob = clf.predict_proba(X_test)
print('[scikit-learn] Predicted probabilities:\n', pred_prob)

pred = clf.predict(X_test)
print('[scikit-learn] Prediction:', pred)
