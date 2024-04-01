#BAGGING ALGORITHMMS
#Bagged Decision Trees
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

kfold = KFold(n_splits=10, shuffle=True, random_state=7)
cart = DecisionTreeClassifier()
num_tree = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_tree, random_state=7)
results = cross_val_score(model, x, y, cv=kfold)
print('CART Accuracy:', results.mean()*100.0)


#RANDOM FOREST
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = RandomForestClassifier(random_state=7, max_features=3, n_estimators=10)
results = cross_val_score(model, x, y, cv=kfold)
print('RF Accuracy:', results.mean()* 100.0)


#Extra Trees Classifier
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = ExtraTreesClassifier(max_features=3, n_estimators=100, random_state=7)
results = cross_val_score(model, x, y, cv=kfold)
print("ETC Accuracy:", results.mean()*100.0)



#BOOSTING ALGORITHM
#Stochatic Gradient Classification
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = GradientBoostingClassifier(n_estimators=100, max_features=3, random_state=7)
results = cross_val_score(model, x, y, cv=kfold)
print('SGC Accuracy:', results.mean()*100.0)

#AdaBoost Classifier
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
model = AdaBoostClassifier(n_estimators=50, learning_rate=1, algorithm='SAMME.R', random_state=None)
results = cross_val_score(model, x, y, cv=kfold)
print('Ada Accuracy:', results.mean()*100.0)


#voting ensembles
kfold = KFold(n_splits=10, shuffle=True, random_state=7)

#create sub models
estimators = []

estimators.append(('Logistic', LogisticRegression(max_iter=1000)))
estimators.append(('cart', DecisionTreeClassifier()))
estimators.append(('svm', SVC()))
estimators.append(('Ada', AdaBoostClassifier()))
estimators.append(('SGC', GradientBoostingClassifier()))

#Create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, x, y, cv=kfold)
print("Ensemble Model:", results.mean()*100.0)

