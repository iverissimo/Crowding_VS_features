
import numpy as np
import os
import os.path as op
import pandas as pd
import utils
import sys

import argparse

import ptitprince as pt # raincloud plots
from visualize import plotting

from behaviour import load_beh_data, manual_responses
from exclusion import CheckData

## path to experiment params
settings_file = op.join(os.getcwd(),'exp_params.yml')

# defined command line options
# this also generates --help and error handling
CLI = argparse.ArgumentParser()

CLI.add_argument("--subject",
                nargs="*",
                type=str,  # any type/callable can be used here
                default=[],
                required=True,
                help = 'Subject number (ex:1). If "all" will run for all participants. If list of subs, will run only those (ex: 1 2 3 4)'
                )

CLI.add_argument("--task",
                #nargs="*",
                type=str,  # any type/callable can be used here
                required=True,
                help = 'What task to look at?\n Options: search, crowding, both'
                )

CLI.add_argument("--cmd",
                #nargs="*",
                type=str,  # any type/callable can be used here
                required=True,
                help = 'What analysis to run?\n Options: search_rt, cs_correlations, etc'
                )

CLI.add_argument("--ses_type",
                #nargs="*",
                type=str,  # any type/callable can be used here
                required=False,
                help = 'Which session type (train/test). Default test'
                )

# parse the command line
args = CLI.parse_args()

# access CLI options
participant = args.subject[0].zfill(3) if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
ses_type = args.ses_type if args.ses_type is not None else 'test' # session to analyze (test vs train)
task = args.task
py_cmd = args.cmd

## load data info etc for participant
data_crowding = load_beh_data.BehCrowding(settings_file, participant, ses_type)
data_search = load_beh_data.BehVsearch(settings_file, participant, ses_type)

## initialize manual responses object for participant
search_behaviour = manual_responses.BehResponses(data_search)
crowding_behaviour = manual_responses.BehResponses(data_crowding)

## initialize plotter objects
crwd_plotter = plotting.PlotsBehavior(crowding_behaviour, outputdir = op.join(data_crowding.derivatives_pth, 
                                                                            'plots', 'crowding'))
search_plotter = plotting.PlotsBehavior(search_behaviour, outputdir = op.join(data_search.derivatives_pth, 
                                                                            'plots', 'search'))

## Run specific command

if task == 'both':

    if py_cmd == 'check_exclusion':

        # check which participants we should exclude
        checker = CheckData(crowding_behaviour, search_behaviour)
        checker.get_exclusion_csv(excl_file = op.join(data_search.derivatives_pth, 'excluded_participants.csv'))

    elif py_cmd == 'correlations_RT':

        # make plot dir, if it doesnt exist
        plot_dir = op.join(data_search.derivatives_pth, 'plots', 'correlations')
        os.makedirs(plot_dir, exist_ok=True)

        # get search RTs for all trials
        search_behaviour.get_RTs(missed_trl_thresh = data_search.params['visual_search']['missed_trl_thresh'])

        # get mean RT and accuracy
        search_behaviour.get_meanRT(df_manual_responses = search_behaviour.df_manual_responses,
                                acc_set_thresh = data_search.params['visual_search']['acc_set_thresh'],
                                acc_total_thresh = data_search.params['visual_search']['acc_total_thresh'])

        # get critical spacing for crowding
        crowding_behaviour.get_critical_spacing(num_trials = data_crowding.nr_trials_flank * data_crowding.ratio_trls_cs,
                                                cs_min_thresh = data_crowding.params['crowding']['cs_min_thresh'],
                                                cs_max_thresh = data_crowding.params['crowding']['cs_max_thresh'])

        ### plot correlations of mean RT with CS
        search_plotter.plot_correlations_RT_CS(df_CS = crowding_behaviour.df_CS, 
                                                df_mean_results = search_behaviour.df_mean_results, 
                                                crowding_type_list = data_crowding.crwd_type,
                                                save_fig = True, outdir = plot_dir)

        # get search slopes
        search_behaviour.get_search_slopes(df_manual_responses = search_behaviour.df_manual_responses)

        ### plot correlations of mean RT with CS
        search_plotter.plot_correlations_slopeRT_CS(df_CS = crowding_behaviour.df_CS, 
                                                    df_search_slopes = search_behaviour.df_search_slopes, 
                                                    crowding_type_list = data_crowding.crwd_type,
                                                    save_fig = True, outdir = plot_dir)


elif task == 'search':
    
    # get RTs for all trials
    search_behaviour.get_RTs(missed_trl_thresh = data_search.params['visual_search']['missed_trl_thresh'])

    if py_cmd == 'RT':
    
        # get mean RT and accuracy
        search_behaviour.get_meanRT(df_manual_responses = search_behaviour.df_manual_responses,
                                acc_set_thresh = data_search.params['visual_search']['acc_set_thresh'],
                                acc_total_thresh = data_search.params['visual_search']['acc_total_thresh'])

        ## get search slopes
        search_behaviour.get_search_slopes(df_manual_responses = search_behaviour.df_manual_responses)

        # plot
        search_plotter.plot_RT_acc_search(df_manual_responses = search_behaviour.df_manual_responses,
                                    df_mean_results = search_behaviour.df_mean_results,
                                    df_search_slopes = search_behaviour.df_search_slopes, save_fig = True)

elif task == 'crowding':

    # get RTs for all trials
    # will produce a manual responses dataframe for all participants selected
    # useful for further analysis
    crowding_behaviour.get_RTs(missed_trl_thresh = data_crowding.params['crowding']['missed_trl_thresh'])   

    if py_cmd == 'RT':

        # get mean RT and accuracy
        crowding_behaviour.get_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses)
        crowding_behaviour.get_NoFlankers_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses, 
                                            acc_thresh = data_crowding.params['crowding']['noflank_acc_thresh'])

        # plot
        crwd_plotter.plot_RT_acc_crowding(df_NoFlanker_results = crowding_behaviour.df_NoFlanker_results, 
                                    no_flank = True, save_fig = True)
        crwd_plotter.plot_RT_acc_crowding(df_manual_responses = crowding_behaviour.df_manual_responses, 
                                    df_mean_results = crowding_behaviour.df_mean_results,
                                    no_flank = False, save_fig = True)

    elif py_cmd == 'CS':

        ## get critical spacing for crowding
        crowding_behaviour.get_critical_spacing(num_trials = data_crowding.nr_trials_flank * data_crowding.ratio_trls_cs,
                                                cs_min_thresh = data_crowding.params['crowding']['cs_min_thresh'],
                                                cs_max_thresh = data_crowding.params['crowding']['cs_max_thresh'])

        # plot
        crwd_plotter.plot_critical_spacing(crowding_behaviour.df_CS, save_fig = True)
        crwd_plotter.plot_staircases(save_fig = True)


# ### STATS ###

# ## check if there's difference between accuracy of different crowding types (includes unflankered)
# utils.repmesANOVA(crowding_behaviour.df_mean_results, 
#                     dep_variable = 'accuracy', within_var = ['crowding_type'], sub_key = 'sj', 
#                     filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'acc_crowding_onewayANOVA.csv'))

# # also do a pair-wise comparison
# utils.wilcox_pairwise_comp(pd.DataFrame({'sj': crowding_behaviour.df_mean_results.sj.values,
#                                         'variable': crowding_behaviour.df_mean_results.crowding_type.values,
#                                         'variable_val': crowding_behaviour.df_mean_results.accuracy.values}),
#                                         crowding_behaviour.df_mean_results.crowding_type.unique(),
#                                         p_value = .001, 
#                                         filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'acc_crowding_pairwiseWilcox.csv'))

# ## check if there's difference between meant RT of different crowding types (includes unflankered)
# utils.repmesANOVA(crowding_behaviour.df_mean_results, 
#                     dep_variable = 'mean_RT', within_var = ['crowding_type'], sub_key = 'sj', 
#                     filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'meanRT_crowding_onewayANOVA.csv'))

# # also do a pair-wise comparison
# utils.wilcox_pairwise_comp(pd.DataFrame({'sj': crowding_behaviour.df_mean_results.sj.values,
#                                         'variable': crowding_behaviour.df_mean_results.crowding_type.values,
#                                         'variable_val': crowding_behaviour.df_mean_results.mean_RT.values}),
#                                         crowding_behaviour.df_mean_results.crowding_type.unique(),
#                                         p_value = .001, 
#                                         filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'meanRT_crowding_pairwiseWilcox.csv'))

# ## check if there's difference between critical spacing of different crowding types 
# utils.repmesANOVA(crowding_behaviour.df_CS, 
#                     dep_variable = 'critical_spacing', within_var = ['crowding_type'], sub_key = 'sj', 
#                     filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'CS_crowding_onewayANOVA.csv'))

# # also do a pair-wise comparison
# utils.wilcox_pairwise_comp(pd.DataFrame({'sj': crowding_behaviour.df_CS.sj.values,
#                                         'variable': crowding_behaviour.df_CS.crowding_type.values,
#                                         'variable_val': crowding_behaviour.df_CS.critical_spacing.values}),
#                                         crowding_behaviour.df_CS.crowding_type.unique(),
#                                         p_value = .001, 
#                                         filename = op.join(crowding_behaviour.dataObj.derivatives_pth,'CS_crowding_pairwiseWilcox.csv'))

# ## two-way repeated-measures ANOVA with set size and eccentricity as factors
# # for search
# utils.repmesANOVA(search_behaviour.df_mean_results, 
#                     dep_variable = 'mean_RT', within_var = ['target_ecc', 'set_size'], sub_key = 'sj', 
#                     filename = op.join(search_behaviour.dataObj.derivatives_pth,'meanRT_Vsearch_twowayANOVA.csv'))
