from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#CLASSIFICATION ACCURACY
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model, x, y, cv=kfold, scoring='accuracy')
print("Classification Accuracy:", result.mean()*100.0, result.std())


#LOGARITHMIC LOSS
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = LogisticRegression(max_iter=1000)
result = cross_val_score(model, x, y, cv=kfold, scoring='neg_log_loss')
print("Logloss:", result.mean(), result.std())


#CLASSIFICATION REPORT
model = LogisticRegression(max_iter=1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
model.fit(x_train, y_train)
predicted = model.predict(x_test)
report = classification_report(y_test, predicted)
print(report)
