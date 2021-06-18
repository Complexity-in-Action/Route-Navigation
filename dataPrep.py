#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lillyrigoli

Included:

- Data Preparation: Filtering, Trimming & Interpolation

- Example code for how to create the three route types

- Code for binning X values within 20 cm bins along y-axis

- Code for 'exploding' nested arrays in dataframe

- Code for converting data format from long to wide (e.g., for repeated measures analysis)

"""

#%% import packages
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

#%% Filter trajectories based on cumulative distance and based on trial length
# ------------------------------
def filterCumDist_trialLength(data):

    filterSize = 1.2
    data['cumDist_avg'] = data.groupby(['obsScenario', 'startPos', 'targetPos'])['cumDist'].transform('mean')
    data = data.loc[data['cumDist'] < data['cumDist_avg']*filterSize]

    # Filter out trials that are longer than 999 rows, i.e., longer than 20 sec
    data = data.loc[data['dataLen'] < 999]

    return data

#%% Interpolate data to N pts
# ------------------------------
def interp_traj(arr, N=1000):

    # interpolate array over `N` evenly spaced points
    min_val = np.min(arr)
    max_val = np.max(arr)

    t_orig = np.linspace(min_val, max_val, len(arr))
    t_interp = np.linspace(min_val, max_val, N)
    f = interp1d(x=t_orig, y=arr)
    interp_arr = f(t_interp)

    return interp_arr

#%% Cut the repeating values in the beginning (from humans not moving @ start of trial for x amount of time. )
# ------------------------------
def startMovement(arr):

    firstMovement = np.where(arr[:-1] != arr[1:])[0]
    firstMovement = firstMovement[0]
    arrLen = len(arr)
    arr = arr[firstMovement:arrLen]
    return arr

#%% Bin data into 20cm bins along 'Y' axis (raw data: Y = X, Z = Y)
# ------------------------------
def binXZdata(data):

    data = pd.DataFrame(data, columns=['X', 'Z'])
    xbins = np.linspace(-16, 16, 161)
    x_cut = pd.cut(data.X, xbins, right=False)
    binned = data.groupby([x_cut]).mean()
    binned.fillna(method='ffill', inplace=True)
    binned.fillna(method='bfill', inplace=True)
    newdf = np.array([xbins[0:160], binned.Z]).T
    return newdf

#%% "Unnest" dataframe if you have nested arrays or similar
# ------------------------------
def unnesting(df, explode):

    idx = df.index.repeat(df[explode[0]].str.len())
    df1 = pd.concat([
        pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
    df1.index = idx

    return df1.join(df.drop(explode, 1), how='right')

#%%  Create Route Types
# ------------------------------
data = pd.read_csv('exampleData.csv')

data['start_target'] = data[['startPos','targetPos']].astype(str).apply(''.join,1)
data['start_target'] = pd.to_numeric(data['start_target'])

conditions = [
    (data['start_target'] == 11) | (data['start_target'] == 23),
    (data['start_target'] == 12) | (data['start_target'] == 22),
    (data['start_target'] == 13) | (data['start_target'] == 21)]

choices = ['same_side', 'middle', 'opp_side']

data['route'] = np.select(conditions, choices, default='NaN')

data_means = data.groupby('route').mean()
data_std = data.groupby('route').std()

#%%
# --------------- GENERAL USE:: CONVERT FROM LONG TO WIDE FORMAT --------------
# Note that 'z-error' here refers to what is called distance in the paper (i.e., binned x-differences)

import pickle
import os
import pandas as pd
import numpy as np

data = pd.read_csv('exampleData.csv')

data['start_target'] = data[['startPos','targetPos']].astype(str).apply(''.join,1)
data['start_target'] = pd.to_numeric(data['start_target'])

conditions = [
    (data['start_target'] == 11) | (data['start_target'] == 23),
    (data['start_target'] == 12) | (data['start_target'] == 22),
    (data['start_target'] == 13) | (data['start_target'] == 21)]

choices = ['same_side', 'middle', 'opp_side']

data['route'] = np.select(conditions, choices, default='NaN')

data = data.groupby(["humanNum", "route"]).mean()
data = data.reset_index()
data = data[['humanNum','route', 'z_error']]

df = data.copy()
df = df.reset_index()
df = df[['humanNum','route', 'z_error']]
df['idx'] = df.groupby('humanNum').cumcount()

tmp = []
for var in ['humanNum','route', 'z_error']:
    df['tmp_idx'] = var + '_' + df.idx.astype(str)
    tmp.append(df.pivot(index='humanNum',columns='tmp_idx',values=var))

reshape = pd.concat(tmp,axis=1)

reshape.rename(columns = {'humanNum_0': 'agentNum', 'z_error_0': 'z_error_middle','z_error_1': 'z_error_oppSide',
                          'z_error_2': 'z_error_sameSide'}, inplace = True)

output = reshape[['agentNum', 'z_error_middle', 'z_error_oppSide', 'z_error_sameSide']]
