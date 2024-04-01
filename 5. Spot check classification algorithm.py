from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#LOGISTIC REGRESSION
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model, x, y, cv=kfold)
print("Accuracy LR:", result.mean()*100)

#KNEAREST NEIGBOUR
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = KNeighborsClassifier()
result = cross_val_score(model, x, y, cv=kfold)
print("Accuracy KNN:", result.mean()*100.0)

#Naive Bayes
kfold = KFold(n_splits=10,  shuffle=True, random_state=7)
model = GaussianNB()
result = cross_val_score(model, x, y, cv=kfold)
print("Accuracy NB: ", result.mean()*100.0)

#CLASSIFICATION AND REGRESSION TREE
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = DecisionTreeClassifier()
result = cross_val_score(model, x, y, cv=kfold)
print("Accuracy C&RT: ", result.mean()*100.0)

#SUPPORT VECTOR MACHINE
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = SVC()
result = cross_val_score(model, x, y, cv=kfold)
print("Accuracy SVM: ", result.mean()*100.0)

