from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
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
import os

path = os.getcwd()+'/datasets/housing/housing.csv'

housing = pd.read_csv(path)

housing.head()

housing.info()
housing.isnull().sum()
# na sur la colonne total_bedrooms, 207
housing.shape
housing['ocean_proximity'].value_counts()
housing.describe()
# test
housing.hist(bins=50, figsize=(15, 15), sharex=False)
# voir la distribution via density plot
housing.plot(kind='density', subplots=True, layout=(3, 3), figsize=(15, 15), sharex=False)
# boites à moustaches
housing.plot(kind='box', subplots=True, sharex=False, layout=(3, 3), figsize=(15, 15))
housing.corr()


# visualisation

df = housing.copy()

# geographiques datas
df.plot(kind='scatter', x='longitude', y='latitude', figsize=(15, 10), alpha=0.5,
        c='median_house_value', colorbar=True, cmap=plt.get_cmap('jet'))

cor = housing.corr()
cor['median_house_value'].sort_values(ascending=False)

mat = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
pd.plotting.scatter_matrix(housing[mat], figsize=(15, 15))


housing = pd.read_csv('~/Bureau/train_ml/datasets/housing/housing.csv')

"""
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)

imputer.fit(housing['total_bedrooms'].values.reshape(-1, 1))
housing['total_bedrooms'] = imputer.transform(housing['total_bedrooms'].values.reshape(-1, 1))

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']
housing.isnull().sum()

housing.corr()['median_house_value'].sort_values(ascending=False)
"""


# division en X/Y
Y = housing['median_house_value']
X = housing.drop(['median_house_value'], axis=1)


class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


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

X_trans = full_pip.fit_transform(X)

num_pip.named_steps['std_scaler']

np.isnan(X_trans).sum()
X_trans.shape

en = OneHotEncoder().fit(X[X_cat])
col = X_num + [i for i in en.categories_[0]]
a = pd.DataFrame(X_trans, columns=col)

a.shape
a.isnull().sum()

X_trans.shape
Y.shape

X_train, X_test, y_train, y_test = train_test_split(X_trans, Y, random_state=0, test_size=0.2)

X_test.shape, y_test.shape
X_train.shape, y_train.shape


models = [
    ('LR', LinearRegression()),
    ('tree', DecisionTreeRegressor()),
    ('ridge', Ridge()),
    ('lasso', Lasso()),
    ('svr', SVR()),
    ('RFR', RandomForestRegressor()),
    ('KNN', KNeighborsRegressor())
]

models
a = np.arange(0, 101, 10)
a
results = []
names = []
names

for name, model in models:
    kfold = KFold(n_splits=2, random_state=0)
    cv_result = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_result.mean())
    names.append(name)


results
a = [np.sqrt(-i) for i in results]
for i, j in zip(names, a):
    print(i, j)


param = {
    'n_estimators': [10, 20, 50, 80, 200, 300, 1000],
    'max_features': [2, 3, 4, 5, 6, 7]
}

kfold = KFold(n_splits=5, random_state=0)
cv_result = cross_val_score(GridSearchCV(RandomForestRegressor(random_state=42), param, verbose=5, n_jobs=-1), X_train, y_train,
                            cv=kfold, scoring='neg_mean_squared_error', verbose=5)

grid = GridSearchCV(RandomForestRegressor(random_state=42, max_features=5), param,
                    scoring='neg_mean_squared_error', cv=2, verbose=5, n_jobs=-1)

grid_result = grid.fit(X_train, y_train)

grid_result.best_params_
grid_result.best_estimator_

cv_result = cross_val_score(grid_result.best_estimator_, X_train, y_train,
                            cv=kfold, scoring='neg_mean_squared_error')

np.sqrt(-cv_result.mean())

final_model = grid_result.best_estimator_

y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

rmse
