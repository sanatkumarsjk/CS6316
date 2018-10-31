import pandas as pd
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

