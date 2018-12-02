# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:34:14 2018

@author: Trey
"""
import pandas as pd
from random import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

###############################################################################
################################## Options ####################################
###############################################################################

# Filename to read data from
DATA_FILENAME = "Clean_Data.csv"

# Whether or not to merge zones into four bigger zones
merge_zones = False

# Whether or not to include a bias term in X
include_bias = False

# Whether to use the entire dataset or a random subset of it
use_entire_dataset = True
ratio_to_use = 0.5

# Whether to train or test
train = True

# If training, whether to use grid search or k-fold cross-validation
use_gridsearch = True
params = {'splitter': ('best', 'random'), 'max_depth': (None, 2, 3)}

# If using cross-validation, number of folds
k = 3

# Initialize classifier below
#clf = LogisticRegression(random_state=42)
clf = DecisionTreeClassifier(random_state = 42)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf = LinearSVC(random_state=42)
#clf = AdaBoostClassifier(dt)

###############################################################################
############################### Implementation ################################
###############################################################################

data = pd.read_csv(DATA_FILENAME)

if not use_entire_dataset:
    indices = data.index.values
    shuffle(indices)
    data = data.loc[indices[:int(len(indices) * ratio_to_use)]]

if merge_zones:
    # Divide zones into four regions
    for index, row in data.iterrows():
        data.at[index, "Zone"] = int(str(data.at[index, "Zone"])[0])

X = data[["Call_Month", "Call_Date", "Call_Time"]]
y = data["Zone"].astype(int)

if include_bias:
    X["bias"] = 1
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train algorithm and print results
if train:
    if use_gridsearch:
        grid = GridSearchCV(clf, params, scoring="accuracy")
        grid.fit(X_train, y_train)
        print("Highest Accuracy:", grid.best_score_)
        print("Best Model:", grid.best_estimator_)
    else:
        print(cross_val_score(clf, X_train, y_train, cv=k, scoring="accuracy"))
else:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))