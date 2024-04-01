from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#LINEAR REGRESSION
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
print("neg_mean_squared_error:", result.mean()*100.0)

#CLASSIFICATION AND REGRESSION TREES
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = DecisionTreeRegressor()
result = cross_val_score(model, x, y, cv=kfold, scoring='neg_mean_squared_error')
print("neg_mean_squared_error:", result.mean()*100.0)
