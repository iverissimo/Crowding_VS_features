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

import itertools

import ptitprince as pt # raincloud plots
from visualize import plotting

from behaviour import load_beh_data, manual_responses

## path to experiment params
settings_file = op.join(os.getcwd(),'exp_params.yml')

## define participant number 
if len(sys.argv) < 2:
    raise NameError('Please specify participant we are looking at (can also be group/all) '
                    'as 1st argument in the command line!')

participant = str(sys.argv[1]).zfill(3) # participant we are looking at
ses_type = 'test' # session to analyze (test vs train)

# list of excluded subs
exclude_sj = ['003', '011', '014', '065']

## load data, info etc for participant
data_crowding = load_beh_data.BehCrowding(settings_file, participant, ses_type, 
                                            exclude_sj = exclude_sj)
data_search = load_beh_data.BehVsearch(settings_file, participant, ses_type, 
                                            exclude_sj = exclude_sj)

## get manual responses for participant
crowding_behaviour = manual_responses.BehResponses(data_crowding)
search_behaviour = manual_responses.BehResponses(data_search)

## get RTs for all trials
crowding_behaviour.get_RTs(missed_trl_thresh = data_crowding.params['crowding']['missed_trl_thresh'])
search_behaviour.get_RTs(missed_trl_thresh = data_crowding.params['visual_search']['missed_trl_thresh'])

## get mean RT and accuracy
crowding_behaviour.get_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses)
crowding_behaviour.get_NoFlankers_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses, 
                                        acc_thresh = data_crowding.params['crowding']['noflank_acc_thresh'])
search_behaviour.get_meanRT(df_manual_responses = search_behaviour.df_manual_responses,
                            acc_set_thresh = data_search.params['visual_search']['acc_set_thresh'],
                            acc_total_thresh = data_search.params['visual_search']['acc_total_thresh'])

## get search slopes
search_behaviour.get_search_slopes(df_manual_responses = search_behaviour.df_manual_responses)

## get critical spacing for crowding
crowding_behaviour.get_critical_spacing(num_trials = data_crowding.nr_trials_flank * data_crowding.ratio_trls_cs,
                                        cs_min_thresh = data_crowding.params['crowding']['cs_min_thresh'],
                                        cs_max_thresh = data_crowding.params['crowding']['cs_max_thresh'])

## update list of excluded subjects
exclude_sj = data_crowding.exclude_sj

for pp in data_crowding.sj_num:
    
    if search_behaviour.exclude_sj_bool['sub-{sj}'.format(sj = pp)] or \
    crowding_behaviour.exclude_sj_bool['sub-{sj}'.format(sj = pp)]:

        if pp not in exclude_sj:
            exclude_sj.append(pp)

## update subject list
data_crowding.sj_num = [sID for sID in data_crowding.sj_num if sID not in exclude_sj]
data_search.sj_num = [sID for sID in data_search.sj_num if sID not in exclude_sj]
    
print('excluding %s participants'%(len(exclude_sj)))
print('total participants left %s '%(len(data_crowding.sj_num)))

# save list in derivatives dir, to use later
np.savetxt(data_crowding.excl_file, np.array(exclude_sj), delimiter=",", fmt='%s')

###### PLOTTING #########
# plots for crowding task
crwd_plotter = plotting.PlotsBehavior(crowding_behaviour)

crwd_plotter.plot_RT_acc_crowding(no_flank = True)
crwd_plotter.plot_RT_acc_crowding(no_flank = False)
crwd_plotter.plot_critical_spacing()
crwd_plotter.plot_staircases()

# plots for search task
search_plotter = plotting.PlotsBehavior(search_behaviour)

search_plotter.plot_RT_acc_search()


### STATS ###

## check if there's difference between accuracy of different crowding types (includes unflankered)
utils.repmesANOVA(crowding_behaviour.df_mean_results, 
                    dep_variable = 'accuracy', within_var = ['crowding_type'], sub_key = 'sj', 
                    filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'acc_crowding_onewayANOVA.csv'))

# also do a pair-wise comparison
utils.wilcox_pairwise_comp(pd.DataFrame({'sj': crowding_behaviour.df_mean_results.sj.values,
                                        'variable': crowding_behaviour.df_mean_results.crowding_type.values,
                                        'variable_val': crowding_behaviour.df_mean_results.accuracy.values}),
                                        crowding_behaviour.df_mean_results.crowding_type.unique(),
                                        p_value = .001, 
                                        filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'acc_crowding_pairwiseWilcox.csv'))

## check if there's difference between meant RT of different crowding types (includes unflankered)
utils.repmesANOVA(crowding_behaviour.df_mean_results, 
                    dep_variable = 'mean_RT', within_var = ['crowding_type'], sub_key = 'sj', 
                    filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'meanRT_crowding_onewayANOVA.csv'))

# also do a pair-wise comparison
utils.wilcox_pairwise_comp(pd.DataFrame({'sj': crowding_behaviour.df_mean_results.sj.values,
                                        'variable': crowding_behaviour.df_mean_results.crowding_type.values,
                                        'variable_val': crowding_behaviour.df_mean_results.mean_RT.values}),
                                        crowding_behaviour.df_mean_results.crowding_type.unique(),
                                        p_value = .001, 
                                        filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'meanRT_crowding_pairwiseWilcox.csv'))

## check if there's difference between critical spacing of different crowding types 
utils.repmesANOVA(crowding_behaviour.df_CS, 
                    dep_variable = 'critical_spacing', within_var = ['crowding_type'], sub_key = 'sj', 
                    filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'CS_crowding_onewayANOVA.csv'))

# also do a pair-wise comparison
utils.wilcox_pairwise_comp(pd.DataFrame({'sj': crowding_behaviour.df_CS.sj.values,
                                        'variable': crowding_behaviour.df_CS.crowding_type.values,
                                        'variable_val': crowding_behaviour.df_CS.critical_spacing.values}),
                                        crowding_behaviour.df_CS.crowding_type.unique(),
                                        p_value = .001, 
                                        filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'CS_crowding_pairwiseWilcox.csv'))

## two-way repeated-measures ANOVA with set size and eccentricity as factors
# for search
utils.repmesANOVA(search_behaviour.df_mean_results, 
                    dep_variable = 'mean_RT', within_var = ['target_ecc', 'set_size'], sub_key = 'sj', 
                    filename = op.join(search_behaviour.dataObj.derivatives_pth,'meanRT_Vsearch_twowayANOVA.csv'))
