from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


filename = "C:\Datasets\pima-indians-diabetes.csv"
dataframe = read_csv(filename)
array = dataframe.values
x = array[:,0:8]
y = array[:,8]

#Create pipeline
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('LDA', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

#Evaluate pipeline
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, x, y, cv=kfold)
print('Accuracy:', results.mean() * 100.0)


#FEATURE EXTRACTION AND MODELING PIPELINE
features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

#create pipeline
estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression(max_iter=1000)))
model = Pipeline(estimators)

#evaluate pipeline
kfold = KFold(n_splits=10, shuffle=True, random_state=7)
results = cross_val_score(model, x, y, cv=kfold)
print('Accuracy:', results.mean() * 100)