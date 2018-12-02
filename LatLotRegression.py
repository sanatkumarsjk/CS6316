import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

###############################################################################
################################## Options ####################################
###############################################################################

# Whether or not to include a bias term in X
include_bias = True

# Whether to train or test
train = True

# If training, whether to use grid-search or cross-validation
use_gridsearch = True
params = {'alpha': [0.001, 0.01, 0.1, 0.5, 1]}

# If using cross-validation, number of folds
k = 3

# Initialize regression algorithm below
#reg = LinearRegression()
reg = Lasso()
#reg = DecisionTreeRegressor()

###############################################################################
############################### Implementation ################################
###############################################################################

data = pd.read_csv("Clean_Data.csv")

# Strip latitude and longitude from data's Location feature
lat_lot = pd.DataFrame(index=data.index, columns=["Latitude", "Longitude"])

rows_to_drop = []
for index, row in data.iterrows():        
    if row["Location"].find('(') == -1:
        rows_to_drop.append(index)
    else:
        lat_lot.at[index, "Latitude"] = float(row["Location"].split('(')[1].split(',')[0])
        lat_lot.at[index, "Longitude"] = float(row["Location"].split('(')[1].split(',')[1][1:-1])

data.drop(["Location"], axis=1, inplace=True)
data.drop(rows_to_drop, axis=0, inplace=True)
lat_lot.dropna(inplace=True)

ss = StandardScaler()
ss.fit_transform(lat_lot)
data = pd.concat([data, lat_lot], axis=1)
    

X = data[["Call_Month", "Call_Date", "Call_Time"]]
y_lat = data["Latitude"]
y_lon = data["Longitude"]

if include_bias:
    X["bias"] = 1

X_lat_train, X_lat_test, y_lat_train, y_lat_test = train_test_split(X, y_lat, test_size=0.2)
X_lon_train, X_lon_test, y_lon_train, y_lon_test = train_test_split(X, y_lon, test_size=0.2)

if train:
    if use_gridsearch:
        grid = GridSearchCV(reg, params, scoring="neg_mean_squared_error")
        grid.fit(X_lat_train, y_lat_train)
        
        print("Best Score (Latitude):", np.sqrt(-grid.best_score_))
        print("Best Model (Latitude):", grid.best_estimator_)
        
        grid.fit(X_lon_train, y_lon_train)
        print("Best Score (Longitude):", np.sqrt(-grid.best_score_))
        print("Best Model (Latitude):", grid.best_estimator_)
    else:
        mse_lat = cross_val_score(reg, X_lat_train, y_lat_train, cv=k, scoring="neg_mean_squared_error")
        mse_lon = cross_val_score(reg, X_lon_train, y_lon_train, cv=k, scoring="neg_mean_squared_error")

        print("Latitude RMSE:", np.sqrt(-mse_lat))
        print("Longitude RMSE:", np.sqrt(-mse_lon))
else:
    reg.fit(X_lat_train, y_lat_train)
    y_lat_pred = reg.predict(X_lat_test)
    mse_lat = mean_squared_error(y_lat_test, y_lat_pred)
    
    reg.fit(X_lon_train, y_lon_train)
    y_lon_pred = reg.predict(X_lon_test)
    mse_lon = mean_squared_error(y_lon_test, y_lon_pred)
    
    print("Latitude RMSE:", np.sqrt(mse_lat))
    print("Longitude RMSE:", np.sqrt(mse_lon))