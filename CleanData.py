import pandas as pd
import numpy as np
import urllib.request

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
def readCsv(filename):
    data = pd.read_csv(filename)
    return data
call_time = readCsv("extractedData.csv")["Call Date/Time"].values

#seperating call_time into date month and time(hours)
date = []
month = []
time = []
def seperateDate(d):
    time.append( d[len(d)- 5 : len(d)-3] )
    date.append(d.split("/", 3)[1])
    month.append(d.split("/",3)[0])
for i in call_time:
    seperateDate(str(i))

#writing the transfromed call time to CSV
def writeCsv():
    data = readCsv("extractedData.csv")
    data['Call_Date'] = date
    data['Call_Month'] = month
    data['Call_Time'] = time
    data.to_csv("transformed.csv")

writeCsv()
