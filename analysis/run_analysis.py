from genericpath import isfile
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils
import sys

import ptitprince as pt # raincloud plots

from behaviour import load_beh_data, manual_responses

## path to experiment params
settings_file = op.join(os.getcwd(),'exp_params.yml')

## define participant number 
if len(sys.argv) < 2:
    raise NameError('Please specify participant we are looking at (can also be group/all) '
                    'as 1st argument in the command line!')

participant = str(sys.argv[1]).zfill(3) # participant we are looking at
ses_type = 'test'

# list of excluded subs
exclude_sj = ['003', '011', '014']

## load data, info etc for participant
data_crowding = load_beh_data.BehCrowding(settings_file, participant, ses_type, 
                                            exclude_sj = exclude_sj)
data_search = load_beh_data.BehVsearch(settings_file, participant, ses_type, 
                                            exclude_sj = exclude_sj)

## get manual responses for participant
crowding_behaviour = manual_responses.BehResponses(data_crowding)
search_behaviour = manual_responses.BehResponses(data_search)

## get RTs for all trials
crowding_behaviour.get_RTs()
search_behaviour.get_RTs()

## get mean RT and accuracy
crowding_behaviour.get_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses)
crowding_behaviour.get_NoFlankers_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses)
search_behaviour.get_meanRT(df_manual_responses = search_behaviour.df_manual_responses)

## get search slopes
search_behaviour.get_search_slopes(df_manual_responses = search_behaviour.df_manual_responses)

## get critical spacing for crowding
crowding_behaviour.get_critical_spacing(num_trials = data_crowding.nr_trials_flank * data_crowding.ratio_trls_cs)

## update list of exlcuded subjects
exclude_sj = data_crowding.exclude_sj

for pp in data_crowding.sj_num:
    
    if search_behaviour.exclude_sj_bool['sub-{sj}'.format(sj = pp)] or \
    crowding_behaviour.exclude_sj_bool['sub-{sj}'.format(sj = pp)]:

        if pp not in exclude_sj:
            exclude_sj.append(pp)
    
print('excluding %s participants'%(len(exclude_sj)))

# save list in derivatives dir, to use later
np.savetxt(data_crowding.excl_file, np.array(exclude_sj), delimiter=",", fmt='%s')

    