import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split

RAW_DATA_FILENAME = "Police_Calls_for_Service.csv"
DATA_URL = "https://data.vbgov.com/api/views/7gep-fpmg/rows.csv?accessType=DOWNLOAD"
CLEAN_DATA_FILENAME = "Clean_Data.csv"


#Load in data, downloading datafile if necessary
try:
    data = pd.read_csv(RAW_DATA_FILENAME)
except:
    # Download file
    response = input(RAW_DATA_FILENAME + " not found. Do you wish to download this file? (Y/N) ")

    if response == "Y":
        print("Downloading...")
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_FILENAME)
        print("Downloaded", RAW_DATA_FILENAME)
        
        data = pd.read_csv(RAW_DATA_FILENAME)
    elif response == "N":
        print("Ending program")
    else:
        print("Unrecognized input. Ending program")

# Drop the features we're not using
print("Dropping unused features")
dropped_features = ["Incident Number", "Subdivision", "Report Number", "Entry Date/Time",
                    "Dispatch Date/Time", "En Route Date/Time", "On Scene Date/Time",
                    "Close Date/Time", "Call Type", "Case Disposition"]
data = data.drop(dropped_features, axis=1)

# Separating call_time into date month and time(hours)
print("Separating Call Date/Time into Date, Month, and Time")
call_time = data["Call Date/Time"].values
date = []
month = []
time = []
year = []
def seperateDate(d):
    #time.append( d[len(d)- 11 : len(d)-9] )
    time.append(d.split(" ")[1].split(":")[0])
    date.append(d.split("/", 3)[1])
    month.append(d.split("/",3)[0])
    year.append(d.split("/", 3)[2].split(" ")[0])
    
for i in call_time:
    seperateDate(str(i))

data = data.drop(["Call Date/Time"], axis = 1)

# Writing the transformed call time to CSV
def writeCsv():
    data['Call_Date'] = date
    data['Call_Month'] = month
    data['Call_Time'] = time
    data['Call_Year'] = year
    data.to_csv("transformed.csv")
writeCsv()

# Drop rows containing incomplete data
# Note: This dataset uses a year of 1899 to indicate a missing date
#print("Dropping rows with incomplete data")
#rows_to_drop = []
#dropped_because_year = 0
#dropped_because_noreport = 0
#for index, row in data.iterrows():
#    if row["Case Disposition"] == "No Report":
#        rows_to_drop.append(index)
#        dropped_because_noreport += 1
#    elif row["Call_Year"] == 1899:
#        rows_to_drop.append(index)
#        dropped_because_year += 1
        
#data = data.drop(rows_to_drop, axis = 0)
#data = data.drop(["Call_Year"], axis = 1) #don't need year anymore
#data.to_csv("transformed_and_dropped.csv")
#
#print("Dropped", len(rows_to_drop), "rows:")
#print("  ", dropped_because_year, " due to year", sep="")
#print("  ", dropped_because_noreport, " due to No Report", sep="")

#Do One Hot Encoding
#print("Performing One Hot Encoding")
#data_prepared = pd.get_dummies(data, columns=["Zone"])
data_prepared = data
data_prepared.to_csv(CLEAN_DATA_FILENAME, index=False)
print("Saved clean data to", CLEAN_DATA_FILENAME)