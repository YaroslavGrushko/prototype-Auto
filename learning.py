
# for array processing
import numpy as np
# to work with .csv/txt/etc.
import pandas as pd
# Import LabelEncoder
from sklearn import preprocessing
# Split the data between the Training Data and Test Data
from sklearn.model_selection import train_test_split
# Classifier:
# kneigbours:
from sklearn.neighbors import KNeighborsClassifier
# Gradient Boosting:
# from sklearn.ensemble import GradientBoostingClassifier
# forest:
# from sklearn.ensemble import RandomForestClassifier
# ada boost:
from sklearn.ensemble import AdaBoostClassifier

# to save model on disk
from joblib import dump, load

# read DB-file with mfccs:
# header = None if dataset has no header
dataset =  pd.read_csv('demo_dataset.csv',sep=',')

# let's encode catigorial data to numbers
dataset_encoded = dataset.copy() 
# creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
dataset_encoded['person']=le.fit_transform(dataset['person'].astype(str))
# let's save LabelEncoder on disk
dump(le, 'LabelEncoder.joblib')

# make 2 datasets X - for input data and Y- for output
#  input data
# X = dataset_encoded.iloc[:, 1:14].values  #for default mfccs parameters
X = dataset_encoded.iloc[:, 1:25].values # for bachloar mfccs parametrs

#  output data
Y = dataset_encoded.iloc[:, 0].values

# CLASSIFIER: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# KNeighbours >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
clf = KNeighborsClassifier(n_neighbors=3)

# << KNeighbours <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
# let's train:
clf.fit(X, Y) 
# xgboost:<<<<<<<<<<<<<<


# let's save model on disk
dump(clf, 'Classifier_model.joblib')
# CLISSIFIER <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

