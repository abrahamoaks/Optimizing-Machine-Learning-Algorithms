from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=7)
model = LogisticRegression()
model.fit(x_train, y_train)

#Save the model to disk
filename = "C:\ML models\diabetes_model.sav"
dump(model, open(filename, 'wb'))

#Load the model from disk
loaded_model = load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result*100)