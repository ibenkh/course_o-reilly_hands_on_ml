from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


housing = pd.read_csv('~/Bureau/train_ml/datasets/housing/housing.csv')

# division en X/Y
Y = housing['median_house_value']
X = housing.drop(['median_house_value'], axis=1)

"""
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values
"""

X_cat = [col for col in list(X) if X[col].dtypes == object]
X_num = [i for i in list(X) if i not in X_cat]

num_pip = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pip = ColumnTransformer([
    ('num', num_pip, X_num),
    ('cat', OneHotEncoder(), X_cat)
])
# division en X/Y
Y = housing['median_house_value']
X = housing.drop(['median_house_value'], axis=1)

X_trans = full_pip.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_trans, Y, random_state=0, test_size=0.2)

X_test.shape, y_test.shape
X_train.shape, y_train.shape

param = [
    {
        'kernel': ['linear'],
        'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]
    },
    {
        'kernel': ['rbf'],
        'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0],
        'gamma': [0.001, 0.01, 0.1, 1]
    }]

grid = GridSearchCV(SVR(), param, scoring='neg_mean_squared_error', verbose=2, n_jobs=6, cv=5)

grid.fit(X_train, y_train)
grid.best_params_

negative_mse = grid.best_score_
rmse = np.sqrt(-negative_mse)
rmse

grid.best_estimator_.coef_

a = np.array([3, 10, 5, 6, 4, 5])
a
np.sort(np.argpartition(a, -5)[-5:])
