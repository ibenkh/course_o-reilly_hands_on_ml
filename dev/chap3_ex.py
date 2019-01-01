from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i)
                                     for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i)
                                    for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)  # fetch_openml() returns targets as strings
    sort_by_target(mnist)  # fetch_openml() returns an unsorted dataset
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')


X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


model = KNeighborsClassifier()  # n_neighbors=4, weights='distance'

param = [{
    'n_neighbors': [4, 5],
    'weights': ["uniform", "distance"]
}]

grid = GridSearchCV(model, param, scoring='accuracy', cv=2, verbose=5, n_jobs=-1)

grid.fit(X_train, y_train)
y_pred = grid.best_estimator_.predict(X_test)

confusion_matrix(y_test, y_pred)
model.score(X_test, y_test)
model.score(X_train, y_train)
a = accuracy_score(y_test, y_pred)
a
grid.best_estimator_

model = LogisticRegression(random_state=42)
cvs = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=3, verbose=5, n_jobs=-1)
cvs
y_pred = model.fit(X_train, y_train).predict(X_test)
accuracy_score(y_test, y_pred)
