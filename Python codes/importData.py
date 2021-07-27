#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lillyrigoli

# Example code for importing all raw data files for humans into master dataframe

"""
# import packages
import os
import numpy as np
import csv
import pandas as pd

# ------------------------------
def import_data(filename): #imports csv data

    with open(filename, newline='') as csvfile:
        d = list(csv.reader(csvfile))[2:]
    dataTemp = []
    data = []
    targetHit = 0

    for row_no in range(len(d)):

        if d[row_no][0]!='P1xpos':
            dataTemp.append([float(i) for i in d[row_no][0:31]])

            if d[row_no][34] != "None" and targetHit==0:
                if d[row_no][34] == "Red Target Object":
                    targetHit = row_no
                else:
                    continue   # ignore unsuccessful trials

        else:
            dataTemp=[]

    if targetHit > 1:

        data = np.array(dataTemp)
        data = data[:,[0,2]]
        data = data[0:targetHit,:]
        dataf = pd.DataFrame(data)
        dataLen = len(dataf)
        x = dataf.iloc[:,0]
        x = x.values.reshape(x.size)
        z =  dataf.iloc[:,1]
        z = z.values.reshape(z.size)
        # Calculate cumulative distance (later used to filter trajectories)
        dx = x[1:]-x[:-1]
        dz = z[1:]-z[:-1]
        step_size = np.sqrt(dx**2+dz**2)
        cumDist = np.concatenate(([0], np.cumsum(step_size)))
        cumDist = cumDist[-1]

    else: # set unsuccessful trial values

        dataTemp = []
        data = 99999
        dataLen = 0
        cumDist = 0

    return dataTemp, data, dataLen, cumDist, targetHit

# ------------------------------
def fn_contents(filename):

    splits=filename.split("_")
    participantNum = int(splits[-8][4:])
    roundNum = int(splits[-7][5:])
    trialNum = int(splits[-6][5:])
    startPos = int(splits[-5][1:])
    targetPos = int(splits[-4])
    obsScenario = int(splits[-3])

    return splits, participantNum, roundNum, trialNum, startPos, targetPos, obsScenario

# ------------------------------
def createAllData(part_dir_human): # input is directory path for human data

    import os
    import numpy as np
    import csv
    import pandas as pd

    trial_list = []
    for file in os.listdir(part_dir_human):
        if file.endswith(".csv"):
            trial_list.append(part_dir_human+"/"+file)

    allData = []
    for trial_no in range(len(trial_list)):

        percentComplete = trial_no/(len(trial_list))
        dataTemp, data, dataLen, cumDist, targetHit = import_data(trial_list[trial_no])

        if cumDist != 0:
            splits, participantNum, roundNum, trialNum, startPos, targetPos, obsScenario = fn_contents_human(trial_list[trial_no])
            print("% complete: ", percentComplete*100, " participant: ", participantNum, " round: ", roundNum, " trialNum: ", trialNum, " startPos: ", startPos, " targetPos: ", targetPos, " obsScenario: ", obsScenario)
            allData.append({'participantNum': participantNum, 'roundNum': roundNum, 'trialNum': trialNum, 'obsScenario': obsScenario, 'startPos': startPos, 'targetPos': targetPos, 'targetHit': targetHit, 'cumDist': cumDist, 'dataLen': dataLen, 'trialTime': dataLen/50, 'data': data})

        else:
            continue

    allData = pd.DataFrame(allData)
    return allData
