from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import SVC


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#PREPARE MODELS
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

#EVALUATE EACH MODEL IN TURN
result = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
    result.append(cv_results)
    names.append(name)
    msg = "%s: %f(%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)

#Boxplot algorithm comparism
fig = plt.figure()
fig.suptitle('Algorithm Comparism')
ax = fig.add_subplot(111)
plt.boxplot(result)
ax.set_xticklabels(names)
plt.show()



