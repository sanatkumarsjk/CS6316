import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split


###############################################################################
#### Modify this section depending on how you want the data cleaned ###########
###############################################################################
DATA_URL = "https://data.vbgov.com/api/views/7gep-fpmg/rows.csv?accessType=DOWNLOAD"
RAW_DATA_FILENAME = "Police_Calls_for_Service.csv"
CLEAN_DATA_FILENAME = "Clean_Data.csv"

# Features that will be dropped from data
dropped_features = [
                    "Incident Number", 
                    "Report Number",
                    "Call Type",
                    "Zone",
                    "Case Disposition",
                    "Priority",
                    "Subdivision",
                    "Call Date/Time',
                    "Entry Date/Time",
                    "Dispatch Date/Time",
                    "En Route Date/Time",
                    "On Scene Date/Time",
                    "Close Date/Time",
                    "Location"
                    ]

# Features that will be One Hot Encoded
cat_features = ["Zone"]

###############################################################################
####################### Do not modify code past this ##########################
###############################################################################

#Load in data, downloading datafile if necessary
try:
    data = pd.read_csv(RAW_DATA_FILENAME)
except:
    # Download file
    print(RAW_DATA_FILENAME + " not found. Downloading file...")
    urllib.request.urlretrieve(DATA_URL, RAW_DATA_FILENAME)
    data = pd.read_csv(RAW_DATA_FILENAME)
    print("File Downloaded")

# Drop the features we're not using
print("Dropping unused features")
data = data.drop(dropped_features, axis=1)

if "Call Date/Time" not in dropped_features:
    # Separating call_time into date month and time(hours)
    print("Separating Call Date/Time into Date, Month, and Time")
    time = []
    date = []
    month = []
    year = []    
    def separateDate(d):
        time.append(d.split(" ")[1].split(":")[0])
        date.append(d.split("/", 3)[1])
        month.append(d.split("/",3)[0])
        year.append(d.split("/", 3)[2].split(" ")[0])
        
    for time in data["Call Date/Time"]:
        separateDate(str(time))
    
    data.drop(["Call Date/Time"], axis=1, inplace=True)
    
    data['Call_Date'] = date
    data['Call_Month'] = month
    data['Call_Time'] = time
    data['Call_Year'] = year

# Drop rows containing incomplete data
# Note: Different features indicate missing values differently in this
# dataset. These only check for missing values for the features that we
# have used in this project.
print("Dropping rows with incomplete data")
rows_to_drop = set()
for index, row in data.iterrows():
    if "Case Disposition" not in dropped_features:
        if row["Case Disposition"] == "No Report":
            rows_to_drop.append(index)
    
    if "Call Date/Time" not in dropped_features:
        if row["Call_Year"] == 1899:
            rows_to_drop.append(index)
                
    if "Priority" not in dropped_features:
        if row["Priority"] == "E":
            rows_to_drop.append(index)

data.drop(rows_to_drop, axis=0, inplace=True)

if "Call Date/Time" not in dropped_features:
    # We only needed year to determine missing dates, so we can drop it now
    data.drop(["Call_Year"], axis = 1, inplace=True)

# Do One Hot Encoding
print("Performing One Hot Encoding")
data = pd.get_dummies(data, columns=cat_features)
data.to_csv(CLEAN_DATA_FILENAME, index=False)
print("Saved clean data to", CLEAN_DATA_FILENAME)