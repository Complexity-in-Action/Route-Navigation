#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lillyrigoli

Included:

- Code for caluclating the confidence intervals and mean trajectory

- Code for caluclating % of mean trajectory within the CIs

"""

# import packages
import numpy as np
import pandas as pd
import scipy as scp

#%% Calulcate CIs and Mean trajectories
#%% ------------------------------
def getConfIntv(data):
    confIntv = pd.DataFrame()
    grps_all = data.groupby(['obsScenario', 'startPos', 'targetPos'])

    dist_all = []

    for group, d in grps_all:
        print("Calculating CI for group: ", str(group))

        xtemp = d[['X', 'participantNum', 'trialNum', 'roundNum']].copy()
        ztemp = d[['Z', 'participantNum', 'trialNum', 'roundNum']].copy()

        xgrps = xtemp.groupby(["participantNum", "trialNum", "roundNum"])
        zgrps = ztemp.groupby(["participantNum", "trialNum", "roundNum"])

        tempdf = pd.DataFrame()
        xdf_all = pd.DataFrame()
        zdf_all = pd.DataFrame()
        i = 0

        for arr, dfx in xgrps:
            if len(dfx) == 160:
                dfz = zgrps.get_group(arr)
                if len(dfz) == 160:
                    xdf_all[i] = np.array(dfx.X)
                    zdf_all[i] = np.array(dfz.Z)
                    i = i+1
        xmean = np.array(xdf_all.mean(axis=1))
        zmean = np.array(zdf_all.mean(axis=1))

        CI_Z = []
        for idx,row in zdf_all.iterrows():
            upper_z = np.percentile(row, 97.5)
            lower_z = np.percentile(row, 2.5)

            CI_Z.append({'mx': xmean[idx], 'mz': zmean[idx], 'upper_z': upper_z, 'lower_z': lower_z, 'group': group})

        CI_Z = pd.DataFrame(CI_Z)
        colMeanCI_name = str(group)+'_mean_Z'
        colTopCI_name = str(group)+'_top_Z'
        colBottomCI_name = str(group)+'_bottom_Z'
        confIntv[colMeanCI_name] = CI_Z.mz
        confIntv[colTopCI_name] = CI_Z.upper_z
        confIntv[colBottomCI_name] = CI_Z.lower_z

    return confIntv

#%% Calulcate % of mean trajectory within the CIs
#%% ------------------------------
def CI_analysis(CI_H_top, CI_H_bottom, CI_R):

    CI_R = CI_R.filter(regex='Z')
    CI_inside = pd.DataFrame()

    c = 0
    for col in CI_R:
        temp = []
        for i, meanR in CI_R[col].iteritems():
            if (meanR < CI_H_top.iloc[i,c]) & (meanR > CI_H_bottom.iloc[i,c]):
                temp.append(1)
            else:
                temp.append(0)
        CI_inside[col] = temp
        c = c+1

    CI_avg = CI_inside.agg([np.average])
    CI_avg = pd.DataFrame(CI_inside.mean(axis=0))
    return CI_avg
