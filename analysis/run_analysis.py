
import numpy as np
import os
import os.path as op
import pandas as pd
import utils
import sys

import argparse

import ptitprince as pt # raincloud plots
from visualize import plotting, eye_plotting

from behaviour import load_beh_data, manual_responses
from exclusion import CheckData
from eyetracking.load_eye_data import EyeTrackVsearch

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
                default='test',
                required=False,
                help = 'Which session type (train/test). Default test'
                )

# parse the command line
args = CLI.parse_args()

# access CLI options
participant = args.subject[0].zfill(3) if len(args.subject) == 1 else args.subject # for situation where 1 sj vs list
ses_type = args.ses_type # session to analyze (test vs train)
task = args.task
py_cmd = args.cmd

## load data info etc for participant
data_crowding = load_beh_data.BehCrowding(settings_file, participant, ses_type)
data_search = load_beh_data.BehVsearch(settings_file, participant, ses_type)

## initialize manual responses object for participant
search_behaviour = manual_responses.BehResponses(data_search)
crowding_behaviour = manual_responses.BehResponses(data_crowding)

## initialize eye tracking object
eye_search = EyeTrackVsearch(data_search)


## initialize plotter objects
crwd_plotter = plotting.PlotsBehavior(crowding_behaviour, outputdir = op.join(data_crowding.derivatives_pth, 
                                                                            'plots', 'crowding'))
search_plotter = plotting.PlotsBehavior(search_behaviour, outputdir = op.join(data_search.derivatives_pth, 
                                                                            'plots', 'search'))

eye_plotter = eye_plotting.PlotsEye(data_search) # plotter for eye data of search

## Run specific command

if task == 'both':

    if py_cmd == 'check_exclusion':

        # check which participants we should exclude
        checker = CheckData(crowding_behaviour, search_behaviour)
        checker.get_exclusion_csv(excl_file = op.join(data_search.derivatives_pth, 'excluded_participants.csv'))

    else:
        # get RTs for all trials
        search_behaviour.get_RTs(missed_trl_thresh = data_search.params['visual_search']['missed_trl_thresh'],
                                exclude_outliers = True, threshold_std = 3)
        
        # get mean RT and accuracy
        search_behaviour.get_meanRT(df_manual_responses = search_behaviour.df_manual_responses,
                                acc_set_thresh = data_search.params['visual_search']['acc_set_thresh'],
                                acc_total_thresh = data_search.params['visual_search']['acc_total_thresh'])
        
        ## get critical spacing for crowding
        crowding_behaviour.get_critical_spacing(num_trials = data_crowding.nr_trials_flank * data_crowding.ratio_trls_cs,
                                                cs_min_thresh = data_crowding.params['crowding']['cs_min_thresh'],
                                                cs_max_thresh = data_crowding.params['crowding']['cs_max_thresh'])

        # make plot dir, if it doesnt exist
        plot_dir = op.join(data_search.derivatives_pth, 'plots', 'correlations')
        os.makedirs(plot_dir, exist_ok=True)

        if py_cmd == 'correlations_RT':

            ## plot correlations of mean RT with CS
            search_plotter.plot_correlations_RT_CS(df_CS = crowding_behaviour.df_CS, 
                                                    df_mean_results = search_behaviour.df_mean_results, 
                                                    crowding_type_list = data_crowding.crwd_type,
                                                    save_fig = True, outdir = plot_dir)

            # get search slopes - collapsing ecc
            search_behaviour.get_search_slopes(df_manual_responses = search_behaviour.df_manual_responses, per_ecc = False)

            ## plot correlations of search slopes with CS
            search_plotter.plot_correlations_slopeRT_CS(df_CS = crowding_behaviour.df_CS, 
                                                        df_search_slopes = search_behaviour.df_search_slopes, 
                                                        crowding_type_list = data_crowding.crwd_type,
                                                        save_fig = True, outdir = plot_dir,
                                                        seed_num = 42)

        elif py_cmd == 'correlations_Fix':

            # get mean number of fixations for search
            eye_search.get_search_mean_fixations(df_manual_responses = search_behaviour.df_manual_responses, exclude_target_fix = True)

            ## plot correlations of Fixations with CS
            eye_plotter.plot_correlations_Fix_CS(df_CS = crowding_behaviour.df_CS, 
                                                df_mean_fixations = eye_search.df_mean_fixations, 
                                                crowding_type_list = data_crowding.crwd_type,
                                                save_fig = True, outdir = plot_dir)

            # get fixations for all trials
            eye_search.get_search_trl_fixations(df_manual_responses = search_behaviour.df_manual_responses, exclude_target_fix = True)

            # get number of fixations per item (slope) - collapsing ecc
            df_search_fix_slopes = eye_search.get_fix_slopes(eye_search.df_trl_fixations.dropna(), fix_nr = True,
                                                            per_ecc = False)

            ## plot correlations of number of fixations per item with CS
            eye_plotter.plot_correlations_slopeNumFix_CS(df_CS = crowding_behaviour.df_CS,  
                                                    df_search_fix_slopes = df_search_fix_slopes, 
                                                    crowding_type_list = data_crowding.crwd_type,
                                                    save_fig = True, outdir = plot_dir,
                                                    seed_num = 24)
            
            ## plot RT-fix rho correlation with CS
            eye_plotter.plot_correlations_RTFixRho_CS(df_CS = crowding_behaviour.df_CS, 
                                                    df_manual_responses = search_behaviour.df_manual_responses, 
                                                    df_trl_fixations = eye_search.df_trl_fixations,
                                                    save_fig = True, outdir = plot_dir, 
                                                    seed_num = 457)

elif task == 'search':
    
    # get RTs for all trials
    search_behaviour.get_RTs(missed_trl_thresh = data_search.params['visual_search']['missed_trl_thresh'],
                            exclude_outliers = True, threshold_std = 3)
    
    # get mean RT and accuracy
    search_behaviour.get_meanRT(df_manual_responses = search_behaviour.df_manual_responses,
                            acc_set_thresh = data_search.params['visual_search']['acc_set_thresh'],
                            acc_total_thresh = data_search.params['visual_search']['acc_total_thresh'])

    if py_cmd in ['RT', 'fix', 'fix_selectivity', 'reliability']:

        if py_cmd == 'reliability':
            
            ## get split-half reliability rho
            rho_sh_RT, df_search_slopes_p1, df_search_slopes_p2 = search_behaviour.calc_RT_split_half_reliability(df_manual_responses = search_behaviour.df_manual_responses, 
                                                                                                                iterations = 1000, seed_num = 29,
                                                                                                                return_slopes_arr = True)
            
            ## plot
            search_plotter.plot_RTreliability(rho_sh_RT = rho_sh_RT, save_fig = True, 
                                              df_search_slopes_p1 = df_search_slopes_p1, df_search_slopes_p2 = df_search_slopes_p2)

        ## get search slopes
        search_behaviour.get_search_slopes(df_manual_responses = search_behaviour.df_manual_responses)

        ## plot
        search_plotter.plot_RT_acc_search(df_manual_responses = search_behaviour.df_manual_responses,
                                    df_mean_results = search_behaviour.df_mean_results,
                                    df_search_slopes = search_behaviour.df_search_slopes, save_fig = True)
        
        if py_cmd == 'fix':

            # get mean number of fixations for search
            eye_search.get_search_mean_fixations(df_manual_responses = search_behaviour.df_manual_responses, exclude_target_fix = True)
            # and for all trials
            eye_search.get_search_trl_fixations(df_manual_responses = search_behaviour.df_manual_responses, exclude_target_fix = True)

            ## plot
            eye_plotter.plot_fixations_search(df_trl_fixations = eye_search.df_trl_fixations,
                                                df_mean_fixations = eye_search.df_mean_fixations, save_fig = True)
        
        elif py_cmd == 'fix_selectivity':
            
            # get all fixations for all trials, labeled by feature 
            df_fixations_on_features = eye_search.get_ALLfix_on_features_df(df_manual_responses = search_behaviour.df_manual_responses,
                                                                            exclude_target_fix = True)
            
            # get ratio of fixations on distractors that share target color
            df_mean_fix_on_DISTfeatures = eye_search.get_mean_fix_on_DTC(df_fixations_on_features = df_fixations_on_features, per_set_size = True)

            ## plot
            eye_plotter.plot_fixDTC_search(df_mean_fix_on_DISTfeatures = df_mean_fix_on_DISTfeatures, save_fig = True)

            # get ratio of fixations on distractors that share target color - aggregated across set size
            df_mean_DTC_ecc = eye_search.get_mean_fix_on_DTC(df_fixations_on_features = df_fixations_on_features, per_set_size = False)

            ## plot correlation between selectivity and efficiency
            eye_plotter.plot_correlations_fixDTC_slopes_search(df_mean_DTC_ecc = df_mean_DTC_ecc, df_search_slopes = search_behaviour.df_search_slopes, 
                                                                seed_num = 846, save_fig = True)

    elif py_cmd == 'scanpath': # plot saccade scan path for participant

        if len(eye_search.dataObj.sj_num) > 1:
            print('WARNING: PLOTTING SCANPATHS OF SEVERAL PARTICIPANTS, might take a while...')

        for pp in eye_search.dataObj.sj_num:
            
            ## get all eye events (fixation and saccades)
            eye_events_df = eye_search.load_pp_eye_events_df(pp, sampling_rate = 1000)

            ## plot
            eye_plotter.plot_participant_scanpath_search(pp, eye_events_df = eye_events_df, save_fig = True) 

elif task == 'crowding':

    # get RTs for all trials
    # will produce a manual responses dataframe for all participants selected
    # useful for further analysis
    crowding_behaviour.get_RTs(missed_trl_thresh = data_crowding.params['crowding']['missed_trl_thresh']) 

    # get mean RT and accuracy
    crowding_behaviour.get_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses)  

    if py_cmd == 'RT':

        ## plot mean RT and accuracy for all crowding conditions
        crwd_plotter.plot_RT_acc_crowding(df_manual_responses = crowding_behaviour.df_manual_responses, 
                                    df_mean_results = crowding_behaviour.df_mean_results,
                                    no_flank = False, save_fig = True)

        ## also plot RT and acc for no-flank condition 
        crowding_behaviour.get_NoFlankers_meanRT(df_manual_responses = crowding_behaviour.df_manual_responses, 
                                            acc_thresh = data_crowding.params['crowding']['noflank_acc_thresh'])

        crwd_plotter.plot_RT_acc_crowding(df_NoFlanker_results = crowding_behaviour.df_NoFlanker_results, 
                                    no_flank = True, save_fig = True)
        
    elif py_cmd in ['CS', 'crwd_estimates', 'CS_corr']:

        ## get critical spacing for crowding
        crowding_behaviour.get_critical_spacing(num_trials = data_crowding.nr_trials_flank * data_crowding.ratio_trls_cs,
                                                cs_min_thresh = data_crowding.params['crowding']['cs_min_thresh'],
                                                cs_max_thresh = data_crowding.params['crowding']['cs_max_thresh'])
        
        if py_cmd == 'CS':

            # plot CS distribution for group
            crwd_plotter.plot_critical_spacing(crowding_behaviour.df_CS, save_fig = True)
            # plot each individual staircase of values
            crwd_plotter.plot_staircases(save_fig = True)

        elif py_cmd == 'crwd_estimates':

            # plot accuracy, split by feature
            crwd_plotter.plot_acc_features_crowding(df_mean_results = crowding_behaviour.df_mean_results, save_fig = True)

            ## get accuracy diff dataframe
            ACC_DIFF = crowding_behaviour.get_feature_acc_diff(df_mean_results = crowding_behaviour.df_mean_results)

            # plot main crowding estimates (CS distribution + delta accuracy feature)
            crwd_plotter.plot_delta_acc_CS(acc_diff_df = ACC_DIFF, df_CS = crowding_behaviour.df_CS, save_fig = True)

        elif py_cmd == 'CS_corr':

            ## plot correlation between CS types
            crwd_plotter.plot_CS_types_correlation(df_CS = crowding_behaviour.df_CS, save_fig = True)


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
