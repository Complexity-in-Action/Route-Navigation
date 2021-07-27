#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lillyrigoli

Fit Fajen & Warren Model to each trajectory. Save resulting parameters (i.e., kg, ko & b)

"""

# import packages
import pickle
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from filter_interpolate import interp_traj
import similaritymeasures
from scipy.optimize import minimize

# Parameters for the FW model
#b = 3.25
#kg = 7.5
c1 = 0.4
c2 = 0.4
#ko = 198
c3 = 6.5
c4 = 0.4
mov_vel = 5
dTime = 0.02

def dx2(x,z,heading,aDx,goalpos,obspos, kg, ko, b):
    obscomp = obs_component(x,z,heading, obspos, ko)
    goaldirmag = np.linalg.norm(goalpos-np.array([x,z]))
    goalangle = math.atan((goalpos[1]-z)/(goalpos[0]-x))-heading
    return (-b * aDx) - ((kg * goalangle) * (math.exp(-c1 * goaldirmag) + c2)) + obscomp

def obs_component(x,z, heading, obs, ko):
    obscomp = 0
    for i in range(0,8):
        obdirmag = np.linalg.norm(obs[2*i:2*i+2]-np.array([x,z]))
        obangle = math.atan((obs[2*i+1]-z)/(obs[2*i]-x))-heading
        obscomp += ko*obangle * math.exp(-c3 * abs(obangle)) * math.exp(-c4*obdirmag)
    return obscomp

def FW_model(start_X, start_Z, start_heading, goalpos, obs, kg, ko, b):

    curr_xpos = start_X
    curr_zpos = start_Z
    curr_hdng = start_heading
    aDx = 0
    targethit = 0
    timesteps = 0
    posx = []
    posz = []
    posx.append(curr_xpos)
    posz.append(curr_zpos)

    while (targethit==0):
        timesteps += 1
        dhdng = dx2(curr_xpos, curr_zpos, curr_hdng, aDx, goalpos, obs, kg, ko, b)
        aDx = dhdng * dTime
        curr_hdng -= aDx
        curr_hdng = (curr_hdng + np.pi) % (2 * np.pi) - np.pi
        curr_xpos += mov_vel * dTime * math.cos(curr_hdng)
        curr_zpos += mov_vel * dTime * math.sin(curr_hdng)
        posx.append(curr_xpos)
        posz.append(curr_zpos)

        if np.linalg.norm(goalpos-np.array([curr_xpos,curr_zpos]))<0.5:
            targethit=1

        elif timesteps > 1000:
            return posx, posz, targethit

    return posx, posz, targethit

def traj_err(k):
    posx, posz, targethit = FW_model(curr_xpos, curr_zpos, curr_hdng, goalpos, obspositions, k[0], k[1], k[2])
    return similaritymeasures.dtw(np.column_stack((posx,posz)), np.column_stack((recorded_trajx, recorded_trajz)))


#%% Apply fitting functions to dataset and save as dataframe
# ------------------------------

obsPos = getObsPositions()

curr_hdng = 0
goalpos_all = [[16,12], [16,0], [16,-12]]

participants = data.groupby('participantNum')

for (part, d) in participants:
    sepTrials = d.groupby(['roundNum', 'trialNum'])
    output = []

    for trial, df in sepTrials:
        tic = time.time()
        curr_hdng = 0
        obspositions = np.asarray(obsPos[(df.obsScenario.iloc[0])])
        recorded_trajx = df.X
        recorded_trajz = df.Z
        curr_xpos = df.X.iloc[0]
        curr_zpos = df.Z.iloc[0]
        goalpos = goalpos_all[(df.targetPos.iloc[0]-1)]
        res = minimize(traj_err,(7.5,198, 3.25),method = 'SLSQP', bounds=((2,40),(50,300),(1,15)))
        posx, posz, targethit = FW_model(curr_xpos, curr_zpos, curr_hdng, goalpos, obspositions, res.x[0], res.x[1], res.x[2])
        posx_interp = interp_traj(posx)
        posz_interp = interp_traj(posz)
        toc = time.time()
        runtime = int(toc-tic)

        output.append({'participantNum': df.participantNum.iloc[0], 'roundNum': df.roundNum.iloc[0], 'trialNum': df.trialNum.iloc[0], 'obsScenario': df.obsScenario.iloc[0], \
                       'startPos': df.startPos.iloc[0], 'targetPos': df.targetPos.iloc[0], 'cumDist': df.cumDist.iloc[0], 'runtime': runtime, 'error': res.fun, 'jac': res.jac, 'success': res.success, \
                           'kg': res.x[0], 'ko': res.x[1], 'b': res.x[2], 'nfev': res.nfev, 'nit': res.nit, 'njev': res.njev})

    csv_fn = str(df.participantNum.iloc[0])+'.csv'
    output = pd.DataFrame(output)
