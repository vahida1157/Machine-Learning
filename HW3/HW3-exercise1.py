from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

# ‘hinge’ is the standard SVM loss (used e.g. by the SVC class)
svc = LinearSVC(loss='hinge')
svc.fit(X_train, Y_train, sample_weight=50)  # 88.07% accuracy.
svc.predict(X_test)
print(svc.score(X_test, Y_test))
