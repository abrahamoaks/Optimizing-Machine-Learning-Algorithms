import numpy as np
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import random

filename = 'C:\Datasets\pima-indians-diabetes.csv'
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

alphas = np.array([1, 0.1, 0.01, 0.001, 0.0001, 0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x,y)

print(grid.best_score_)
print(grid.best_estimator_)

#Random Search Parameter Tuning
param_grid = {'alpha': random.uniform()}
model = Ridge()
rsearch = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, scoring=None, random_state=7)
rsearch.fit(x,y)

print(rsearch.best_score_)
print(rsearch.best_estimator_.alpha)