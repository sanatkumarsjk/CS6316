# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:34:14 2018

@author: Trey
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
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
merge_zones = True

# Whether or not to include a bias term in X
include_bias = True

# Whether to do k-fold cross-validation on training data or test on test data
train = True
k = 3

# Initialize classifier below
#clf = DecisionTreeClassifier(random_state = 42)
clf = LogisticRegression(random_state=42)
#clf = RandomForestClassifier(n_estimators=100, random_state=42)
#clf = AdaBoostClassifier(dt)

###############################################################################
############################### Implementation ################################
###############################################################################

data = pd.read_csv(DATA_FILENAME)

if merge_zones:
    # Divide zones into four regions
    for index, row in data.iterrows():
        data.at[index, "Zone"] = int(str(data.at[index, "Zone"])[0])
    

## Divide dataset to make training more tractable
#indices = list(data.index.values)
#random.shuffle(indices)
#
#small_data = data.loc[indices[:len(indices) // 4]]
#
#X = small_data[["Call_Date", "Call_Month", "Call_Time"]]
#y = small_data["Zone"]
#y = y.astype(int)
#

X = data[["Call_Time"]]
y = data["Zone"].astype(int)

#X = pd.get_dummies(X, columns=["Call_Time"])

if include_bias:
    X["bias"] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train algorithm and print results
if train:
    print(cross_val_score(clf, X_train, y_train, cv=k, scoring="accuracy"))
else:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_train, y_pred))

#print("Fitting Classifier...")
#clf.fit(X_train, y_train)
#print("Fit Classifier")
#y_pred = clf.predict(X_train)
#print(accuracy_score(y_train, y_pred))