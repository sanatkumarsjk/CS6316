import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

###############################################################################
################################## Options ####################################
###############################################################################

# Whether or not data preprocessing needs to be done (mostly stripping
# latitude and longitude from the Location feature). If the data is already
# saved in the working environment, setting this to False can save time
DATA_PREPROCESSING = True

model_type = "linear"

# Whether or not to include a bias term in X
include_bias = True

# Whether to do k-fold cross-validation on training data or test on test data
train = True
k = 3

###############################################################################
############################### Implementation ################################
###############################################################################

data = pd.read_csv("Clean_Data.csv")

if DATA_PREPROCESSING:   
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

if model_type == "linear":
    if train:
        lr = LinearRegression()
        mse_lat = cross_val_score(lr, X_lat_train, y_lat_train, cv=k, scoring="neg_mean_squared_error")
        
        lr = LinearRegression()
        mse_lon = cross_val_score(lr, X_lon_train, y_lon_train, cv=k, scoring="neg_mean_squared_error")
        
        print("Latitude RMSE:", np.sqrt(-mse_lat))
        print("Longitude RMSE:", np.sqrt(-mse_lon))
    else:
        lr= LinearRegression()
        lr.fit(X_lat_train, y_lat_train)
        y_lat_pred = lr.predict(X_lat_test)
        mse_lat = mean_squared_error(y_lat_test, y_lat_pred)
        
        lr = LinearRegression()
        lr.fit(X_lon_train, y_lon_train)
        y_lon_pred = lr.predict(X_lon_test)
        mse_lon = mean_squared_error(y_lon_test, y_lon_pred)
        
        print("Linear Model")
        print('-'*30)
        print("Latitude RMSE:", np.sqrt(mse_lat))
        print("Longitude RMSE:", np.sqrt(mse_lon))
    
elif model_type == "decision_tree":
    if train:
        dtr = DecisionTreeRegressor()
        mse_lat = cross_val_score(dtr, X_lat_train, y_lat_train, cv=k, scoring="neg_mean_squared_error")
        
        dtr = DecisionTreeRegressor()
        mse_lon = cross_val_score(dtr, X_lon_train, y_lon_train, cv=k, scoring="neg_mean_squared_error")
        
        print("Latitude RMSE:", np.sqrt(-mse_lat))
        print("Longitude RMSE:", np.sqrt(-mse_lon))
    else:
        dtr = DecisionTreeRegressor()
        dtr.fit(X_lat_train, y_lat_train)
        y_lat_pred = dtr.predict(X_lat_test)
        mse_lat = mean_squared_error(y_lat_test, y_lat_pred)
        
        dtr = DecisionTreeRegressor()
        dtr.fit(X_lon_train, y_lon_train)
        y_lon_pred = dtr.predict(X_lon_test)
        mse_lon = mean_squared_error(y_lon_test, y_lon_pred)
        
        print("Decision Tree")
        print('-'*30)
        print("Latitude RMSE:", np.sqrt(mse_lat))
        print("Longitude RMSE:", np.sqrt(mse_lon))