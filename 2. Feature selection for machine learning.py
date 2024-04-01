from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values

x = array[:,0:8]
y = array[:,8]

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(x,y)

#summarize score
set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(x)

#summarize selected features
print(features[0:,5:])
