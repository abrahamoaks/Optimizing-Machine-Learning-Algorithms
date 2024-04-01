#Load Libraries

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier


#Load Dataset
filename = "C:\Datasets\sonar.csv"
dataset = read_csv(filename, header=None)


#Shape
print(dataset.shape)

#Type
set_option('display.max_rows',500)
print(dataset.dtypes)

#Head
set_option('display.width',100)
print(dataset.head(21))

#Description
set_option('precision', 3)
print(dataset.describe())

#Histogram
dataset.hist(sharex=False, sharey=False)
plt.show()

#Density
dataset.plot(kind='density', subplot=True, sharex=False, legend=False, fontsize=1)
plt.show()


        #VALIDATION DATASET
#split-out validation dataset
array = dataset.values
x = array[:,0:60]
y = array[:,60]

validation_size = 0.20
seed = 7

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_size, random_state=seed)

    #Evaluate Algorithms: Baseline
#Test options and evaluation metrics


#Spot-Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Algorithm Compare')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#EVALUATE ALGORITHMS: Standardize Data
#Standardize the Dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler',StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler',StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler',StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler',StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler',StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler',StandardScaler()), ('SVM', SVC())])))

results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Compare Algorithms
fig = pyplot.figure()
fig.suptitle('Scaled Algorithm Comparison')
pyplot.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Tuning SVM
scaler = StandardScaler().fit(x_train)
rescaledX = scaler.transform(x_train)
C_values = np.array[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(c=C_values, kernel=kernel_values, model=SVC())
kfold = KFold(n_splits=10, random_state=7)
model = SVC()
grid = GridSearchCV(svm.SVC(gama='auto'), {'C': [1,10,20]})


for mean, stdev, param in zip (means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))