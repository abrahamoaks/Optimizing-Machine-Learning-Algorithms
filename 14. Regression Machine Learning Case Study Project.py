#Load Libraries

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


#Load Dataset
filename = "C:\Datasets\housing.csv"
dataset = read_csv(filename)


#Analyze Data with Descriptive Statistics
print(dataset.shape)
print(dataset.head(10))

set_option('precision', 1)
print(dataset.describe())

#Correlation
set_option('precision', 2)
print(dataset.corr(method='pearson'))

                                       # DATA VISUALIZATION
#Histogram
dataset.hist(sharey=False, sharex=False)
plt.show()

#Density plot
dataset.plot(kind='density', layout=(4,4), sharex=False, sharey=False, fontsize=8)
plt.show()

#scatter plot matrix
scatter_matrix(dataset)
plt.show()

#Corellation Matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)
ticks = np.arange(0,14,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xtickslabel(names)
ax.set_ytickslabel(names)
plt.show()


                                #VALIDATION DATASET
array = dataset.values
x = array[:,0:13]
y = array[:,13]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=7)

                                #EVALUATE ALGORITHMS BASELINE
# Test options and evaluation Metric
num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'


                                #SPOT CHECK ALGORITHM
models = []
models.append(('LR', LogisticRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)


                        #COMPARE ALGORITHMS
fig = pyplot.figure()
fig.suptitle('Algorithm Comparism')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.setxticks(names)
plt.show()

                    #EVALUATE ALGORITHMS: STANDARDIZATION
#Standardize the dataset
pipelines = []
pipelines.append(('scaledLR', Pipeline([('scaler',StandardScaler(), ('LR', LogisticRegression())])))
pipelines.append(('scaledLASSO', Pipeline([('scaler',StandardScaler(), ('LASSO',Lasso())])))
pipelines.append(('scaledEN', Pipeline([('scaler',StandardScaler(), ('EN', ElasticNet())])))
pipelines.append(('scaledKNN', Pipeline([('scaler',StandardScaler(), ('KNN', KNeighborsRegressor())])))
pipelines.append(('scaledCART', Pipeline([('scaler',StandardScaler(), ('CART', DecisionTreeRegressor())])))
pipelines.append(('scaledSVR', Pipeline([('scaler',StandardScaler(), ('SVR', SVR())])))

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler',StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler',StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler',StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler',StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler',StandardScaler()), ('EN', ElasticNet())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler',StandardScaler()), ('SVM', SVC())])))


results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)

    msg = "%s %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)