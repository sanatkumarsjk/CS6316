import pandas as pd
import numpy as np
import urllib.request
from sklearn.model_selection import train_test_split

DATA_FILENAME = "Police_Calls_for_Service.csv"
DATA_URL = "https://data.vbgov.com/api/views/7gep-fpmg/rows.csv?accessType=DOWNLOAD"

#Load in data, downloading datafile if necessary
try:
    data = pd.read_csv(DATA_FILENAME)
except:
    # Download file
    response = input(DATA_FILENAME + " not found. Do you wish to download this file? (Y/N) ")

    if response == "Y":
        print("Downloading...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILENAME)
        print("Downloaded", DATA_FILENAME)
    elif response == "N":
        print("Ending program")
    else:
        print("Unrecognized input. Ending program")

# Drop the features we're not using
dropped_features = ["Incident Number", "Report Number", "Subdivision", "Entry Date/Time",
                    "Dispatch Date/Time", "En Route Date/Time", "On Scene Date/Time",
                    "Close Date/Time", "Location"]
data = data.drop(dropped_features, axis=1)

#reading specific column in CSV
#def readCsv(filename):
#    data = pd.read_csv(filename)
#    return data
#call_time = readCsv("extractedData.csv")["Call Date/Time"].values
call_time = data["Call Date/Time"].values

#seperating call_time into date month and time(hours)
date = []
month = []
time = []
year = []
def seperateDate(d):
    time.append( d[len(d)- 5 : len(d)-3] )
    date.append(d.split("/", 3)[1])
    month.append(d.split("/",3)[0])
    year.append(d.split("/", 3)[2].split(" ")[0])
    
for i in call_time:
    seperateDate(str(i))

#writing the transfromed call time to CSV
def writeCsv():
    #data = readCsv("extractedData.csv")
    data['Call_Date'] = date
    data['Call_Month'] = month
    data['Call_Time'] = time
    data['Call_Year'] = year
    data.to_csv("transformed.csv")
writeCsv()

# Drop rows containing incomplete data
rows_to_drop = []
dropped_because_year = 0
dropped_because_noreport = 0
for index, row in data.iterrows():
    if row["Case Disposition"] == "No Report":
        rows_to_drop.append(index)
        dropped_because_noreport += 1
    elif row["Call_Year"] == 1899:
        rows_to_drop.append(index)
        dropped_because_year += 1
        
data = data.drop(rows_to_drop, axis=0)
data.to_csv("transformed_and_dropped.csv")
print("Dropped", len(rows_to_drop), "rows:")
print("  ", dropped_because_year, " due to year", sep="")
print("  ", dropped_because_noreport, " due to No Report", sep="")

# TODO: Add One Hot Encoding for Categorical Data
