from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values

x = array[:,0:8]
y = array[:,8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
print("LR Accuracy:",  result*100.0)

#K-FOLD CROSS VALIDATION
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(max_iter=1000)
results = cross_val_score(model, x, y, cv=kfold)
print("LR KF Accuracy:",  (results.mean()*100.0, results.std()*100.0))
