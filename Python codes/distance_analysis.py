#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lillyrigoli

Example code for computing distance measures

Two examples are included:
- Within distance
- Between distance

"""

# import packages
import pickle
import os
import pandas as pd
import numpy as np

#%% Within Distance (i.e., within humans or within agents)
# ----------------------------------------------------------------------------------
# --- (X-Diff) Compute Raw X Error Between all Human *OR* all Agent trajectories ---
# ----------------------------------------------------------------------------------

pickleFile = open('rawData_RC.pickle', 'rb')
rawData_RC = pickle.load(pickleFile)

err_df = []
sepTrials = rawData_RC.groupby(['participantNum','obsScenario', 'startPos', 'targetPos', 'roundNum', 'trialNum'])

for grp, df in sepTrials:
    for grp2, df2 in sepTrials:

        if len(df) == 160 and len(df2) == 160 and grp[0] != grp2[0] and grp[1:4] == grp2[1:4]:
            err = abs(np.array(df.Z) - np.array(df2.Z))
            err = err.sum()
            err_df.append({'participantNum_1': df.participantNum.iloc[0],'participantNum_2': df2.participantNum.iloc[0], 'obsScenario': df.obsScenario.iloc[0], \
                             'startPos': df.startPos.iloc[0], 'targetPos': df.targetPos.iloc[0], 'err_z': err})

err_df = pd.DataFrame(err_df)
err_Z_summary = err_df.groupby(["obsScenario","startPos", "targetPos"]).agg((min, max, np.nanmean, np.nanstd))

with open ('error_Z_raycast.pickle', 'wb') as handle:
    pickle.dump(err_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%% Between Distance (i.e., comparing humans to agents)
# ----------------------------------------------------------------------------------
# --------------- (X-Diff) Compare Raw Human to Agent Trajectories --------------
# ----------------------------------------------------------------------------------

pickleFile = open('rawData_RC.pickle', 'rb')
rawData_RC = pickle.load(pickleFile)
pickleFile = open('rawData_human.pickle', 'rb')
rawData_human = pickle.load(pickleFile)

humanTrials = rawData_human.groupby(['participantNum', 'obsScenario', 'startPos', 'targetPos'])
agentTrials = rawData_RC.groupby(['participantNum', 'obsScenario', 'startPos', 'targetPos'])
human_RCagent_err = []
i=0

for humanGroup, humanDF in humanTrials:
    currGroup = humanGroup[1:]

    for agent in range(1,21):
        currAgent = 'AIRC'+str(agent)
        currAgentGroup = (currAgent,) + currGroup

        if currAgentGroup in agentTrials.groups.keys():
            agentDF = agentTrials.get_group(currAgentGroup)
            htrials = humanDF.groupby(['roundNum','trialNum'])
            atrials = agentDF.groupby(['roundNum','trialNum'])

            for htrial, hdf in htrials:
                for atrial, adf in atrials:
                    if len(hdf) == 160 and len(adf) == 160:
                        err = abs(np.array(hdf.Z) - np.array(adf.Z))
                        err = err.sum()

                        human_RCagent_err.append({'humanNum': humanDF.participantNum.iloc[0], 'agentNum': currAgent, 'obsScenario': humanDF.obsScenario.iloc[0], \
                                  'startPos': humanDF.startPos.iloc[0], 'targetPos': humanDF.targetPos.iloc[0],'human_roundNum': hdf.roundNum.iloc[0],\
                                  'human_trialNum': hdf.trialNum.iloc[0], 'agent_roundNum': adf.roundNum.iloc[0], 'agent_trialNum': adf.trialNum.iloc[0],'z_error': err})

human_RCagent_Z_raw_err = pd.DataFrame(human_RCagent_err)

human_RCagent_raw_err_Z_summary = human_RCagent_Z_raw_err.groupby(["obsScenario","startPos", "targetPos"]).agg((min, max, np.nanmean, np.nanstd))


with open ('human_RCagent_Z_raw_err.pickle', 'wb') as handle:
    pickle.dump(human_RCagent_Z_raw_err, handle, protocol=pickle.HIGHEST_PROTOCOL)
