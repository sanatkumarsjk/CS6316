import pandas as pd
import numpy as np

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

