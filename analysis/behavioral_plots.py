import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils

import ptitprince as pt # raincloud plots

# load settings from yaml
with open(op.join(os.getcwd(),'exp_params.yml'), 'r') as f_in:
    params = yaml.safe_load(f_in)

# session type and task name
ses_type = 'test'
tasks = ['Crowding']#, 'VSearch'] 

# some params
distance_bounds = params['crowding']['staircase']['distance_ratio_bounds']
crwd_type = params['crowding']['crwd_type'] # types of crowding
ratio_trls_cs = 1/2 # ration of trials to use for CS calculation

# some paths
data_path = params['paths']['data_pth'][params['paths']['curr_dir']]
derivatives_pth = op.join(data_path,'derivatives')
sourcedata_pth = op.join(data_path,'sourcedata')
plot_path = {'Crowding': op.join(derivatives_pth, 'plots', 'Crowding'),
             'VSearch': op.join(derivatives_pth, 'plots', 'VSearch')}

# excluded subs
excl_sub = []#['sub-003', 'sub-011']
# subject list
sub_list = [s for s in os.listdir(sourcedata_pth) if 'sub-' in s and s not in excl_sub]


for tsk in tasks:

    # loop over subjects
    for ind, sj in enumerate(sub_list):

        if not op.isdir(plot_path[tsk]):
            os.makedirs(plot_path[tsk])

        # subject data folder
        sj_pth = op.join(sourcedata_pth, '{sj}'.format(sj=sj))

        # load events files, with all events and sub responses
        events_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('{sj}_ses-{ses}'.format(sj = sj, ses = ses_type)) and \
                        'task-{task}'.format(task = tsk) in x and x.endswith('_events.tsv')]
        events_df = pd.read_csv(events_files[0], sep="\t")
        # only select onsets > 0 (rest is intruction time)
        events_df = events_df[events_df['onset']>0]

        # load trial info, with step up for that session
        trial_info_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('{sj}_ses-{ses}'.format(sj = sj, ses = ses_type)) and \
                        'task-{task}'.format(task = tsk) in x and x.endswith('_trial_info.csv')]
        trial_info_df = pd.read_csv(trial_info_files[0])

        # number of trials
        nr_trials_flank = len(trial_info_df.loc[trial_info_df['crowding_type']=='orientation']['index'].values)
        nr_trials_unflank = len(trial_info_df.loc[trial_info_df['crowding_type']=='unflankered']['index'].values)
            
        # get accuracy unflankered
        df_acc_unflankered, df_rt_unflankered = utils.get_accuracy_rt_uncrowded(events_df, trial_info_df, 
                                                                        acc_type = ['total', 'color', 'orientation'],
                                                                        target_keys = params['keys']['target_key'], sj = sj)

        # append for future use
        if ind == 0:
            df_uncrwd_group = df_acc_unflankered 
        else:
            df_uncrwd_group = pd.concat([df_uncrwd_group, df_acc_unflankered])

        #### plot reaction times and accuracy for unflankered trials ####
        # for single subject
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

        pt.RainCloud(data = df_rt_unflankered, x = 'type', y = 'RT', pointplot = True, 
                    linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
        ax1.set_ylabel('RT (seconds)')
        ax1.set_xlabel('')
        ax1.set_title('Reaction Times crowding (without flankers)')
        ax1.set_ylim(0.2,1.5)

        pt.RainCloud(data = df_acc_unflankered, x = 'type', y = 'accuracy', pointplot = True, 
                    linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
        #sns.boxplot(x = 'type', y = 'accuracy', data = df_acc_unflankered, ax = ax3)
        #sns.swarmplot(x = 'type', y = 'accuracy', data = df_acc_unflankered, color=".3", ax = ax3)
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('')
        ax2.set_title('Accuracy crowding (without flankers)')
        ax2.set_ylim(0,1.05)

        fig.savefig(op.join(plot_path[tsk], '{sj}_ses-{ses}_task-{task}_RT_ACC_UNflankered.png'.format(sj = sj, ses = ses_type, task = tsk)))
        ########

        ## load staircases
        # absolute path for staircase files
        staircase_files =  [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('{sj}_ses-{ses}'.format(sj = sj, ses = ses_type)) and \
                        'task-{task}'.format(task = tsk) in x and 'staircase' in x and x.endswith('.pickle')]
        # load each staircase, save in dict for ease
        staircases = {}
        for crwd in crwd_type:
            staircases[crwd] = pd.read_pickle([val for val in staircase_files if crwd in val][0]).intensities


        ### plot distance ratio per crowding type for visual inspection##
        fig, axis = plt.subplots(1,figsize=(10,5),dpi=100)

        sns.lineplot(data = staircases, drawstyle='steps-pre')
        plt.xlabel('# Trials',fontsize=10)
        plt.ylabel('Distance ratio',fontsize=10)
        plt.title('Staircase for each crowding type',fontsize=10)
        plt.legend(loc = 'upper right',fontsize=7)
        plt.xlim([0, nr_trials_flank])
        plt.ylim([distance_bounds[0]-.05, distance_bounds[-1]+.05])
        fig.savefig(op.join(plot_path[tsk], '{sj}_ses-{ses}_task-{task}_staircases.png'.format(sj = sj, ses = ses_type, task = tsk)))

        #### get df of behavioral results for each crowidng type ###
        
        df_rt = {'crowding_type': [], 'RT': []} # for individual RT plot, checking distribution of trials

        for w, crwd in enumerate(crwd_type):
            # get correct responses
            response_bool, trials = utils.get_correct_responses_crowding(events_df, trial_info_df, 
                                                            crwd_type = crwd,
                                                            target_keys = params['keys']['target_key'])
            # calculate accuracy
            acc = sum(response_bool)/len(response_bool)
            
            # get reaction time for trials of correct response
            rts = events_df[(events_df['trial_nr'].isin(trials[response_bool])) & \
                    (events_df['event_type'] == 'response')].drop_duplicates(subset=['trial_nr'], keep='first')['onset'].values - \
                events_df[(events_df['trial_nr'].isin(trials[response_bool])) & \
                    (events_df['event_type'] == 'stim')]['onset'].values 
            
            df_rt['crowding_type'].append(crwd)
            df_rt['RT'].append(rts)

            # save all in dataframe
            if (w == 0) and (ind == 0):
                df_crwd_results = pd.DataFrame({'sj': [sj],
                                                    'crowding_type': [crwd],
                                                    'accuracy': [acc],
                                                    'mean_RT': [np.nanmean(rts)],
                                                    'critical_spacing': [np.mean(staircases[crwd][-int(nr_trials_flank*ratio_trls_cs):])]})
            else:
                df_crwd_results = pd.concat([df_crwd_results, 
                                                pd.DataFrame({'sj': [sj],
                                                    'crowding_type': [crwd],
                                                    'accuracy': [acc],
                                                    'mean_RT': [np.nanmean(rts)],
                                                    'critical_spacing': [np.mean(staircases[crwd][-int(nr_trials_flank*ratio_trls_cs):])]})
                                                ])
            
            
        ## append uncrowded values on same dataframe
        df_crwd_results = pd.concat([df_crwd_results, 
                                        pd.DataFrame({'sj': [sj],
                                                    'crowding_type': ['unflankered'],
                                                    'accuracy': [df_acc_unflankered[df_acc_unflankered['type']=='total'].accuracy[0]],
                                                    'mean_RT': [df_acc_unflankered[df_acc_unflankered['type']=='total'].mean_RT[0]],
                                                    'critical_spacing': [np.nan]})
                                                ])

        df_rt['crowding_type'].append('unflankered')
        df_rt['RT'].append(list(df_rt_unflankered[df_rt_unflankered['type']=='total'].RT.values))

        df_rt = pd.DataFrame.from_dict(df_rt).set_index(['crowding_type']).apply(pd.Series.explode).reset_index()
        df_rt['RT'] = df_rt['RT'].astype(float)

        #### plot reaction times and accuracy for unflankered trials ####
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

        pt.RainCloud(data = df_rt, x = 'crowding_type', y = 'RT', pointplot = True, 
                    linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
        ax1.set_ylabel('RT (seconds)')
        ax1.set_xlabel('')
        ax1.set_title('Reaction Times crowding')
        ax1.set_ylim(0.2,1.5)

        pt.RainCloud(data = df_crwd_results[df_crwd_results['sj'] == sj], x = 'crowding_type', y = 'accuracy', pointplot = True, 
                    linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
        #sns.boxplot(x = 'type', y = 'accuracy', data = df_acc_unflankered, ax = ax3)
        #sns.swarmplot(x = 'type', y = 'accuracy', data = df_acc_unflankered, color=".3", ax = ax3)
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('')
        ax2.set_title('Accuracy crowding')
        ax2.set_ylim(0,1.05)

        fig.savefig(op.join(plot_path[tsk], '{sj}_ses-{ses}_task-{task}_RT_ACC_flankered.png'.format(sj = sj, ses = ses_type, task = tsk)))
        ########

    
    ## GROUP PLOT ## - uncrowded
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10), dpi=100, facecolor='w', edgecolor='k')

    pt.RainCloud(data = df_uncrwd_group, x = 'type', y = 'mean_RT', pointplot = True, 
                linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
    ax1.set_ylabel('Mean RT (s)')
    ax1.set_xlabel('')
    ax1.set_title('Mean Reaction Time crowding (without flankers)')
    ax1.set_ylim(0.2,1.5)

    pt.RainCloud(data = df_uncrwd_group, x = 'type', y = 'accuracy', pointplot = True, 
                linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('')
    ax2.set_title('Accuracy crowding (without flankers)')
    ax2.set_ylim(0,1.05)

    sns.lineplot(x = 'type', y = 'value', hue = 'sj', 
                data = pd.melt(df_uncrwd_group[["sj", "type", "mean_RT"]], ['type', 'sj']), ax = ax3)
    ax3.set_ylabel('Mean RT (s)')
    ax3.set_xlabel('')
    ax3.set_title('Mean Reaction Time crowding (without flankers)')
    ax3.set_ylim(0.2,1.5)
    ax3.legend(fontsize=8, loc='upper right')

    sns.lineplot(x = 'type', y = 'value', hue = 'sj', 
                data = pd.melt(df_uncrwd_group[["sj", "type", "accuracy"]], ['type', 'sj']), ax = ax4)
    ax4.set_ylabel('Accuracy')
    ax4.set_xlabel('')
    ax4.set_title('Accuracy crowding (without flankers)')
    ax4.set_ylim(0,1.05)
    ax4.legend(fontsize=8, loc='lower right')

    fig.savefig(op.join(plot_path[tsk], 'sj-all_ses-{ses}_task-{task}_mean_RT_ACC_UNflankered.png'.format(ses = ses_type, task = tsk)))

    ## GROUP PLOT ## - crowded
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10), dpi=100, facecolor='w', edgecolor='k')

    pt.RainCloud(data = df_crwd_results, x = 'crowding_type', y = 'mean_RT', pointplot = True, 
                linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
    ax1.set_ylabel('Mean RT (s)')
    ax1.set_xlabel('')
    ax1.set_title('Mean Reaction Time crowding')
    ax1.set_ylim(0.2,1.5)

    pt.RainCloud(data = df_crwd_results, x = 'crowding_type', y = 'accuracy', pointplot = True, 
                linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('')
    ax2.set_title('Accuracy crowding')
    ax2.set_ylim(0,1.05)

    sns.lineplot(x = 'crowding_type', y = 'value', hue = 'sj', 
                data = pd.melt(df_crwd_results[["sj", "crowding_type", "mean_RT"]], ['crowding_type', 'sj']), ax = ax3)
    ax3.set_ylabel('Mean RT (s)')
    ax3.set_xlabel('')
    ax3.set_title('Mean Reaction Time crowding (without flankers)')
    ax3.set_ylim(0.2,1.5)
    ax3.legend(fontsize=8, loc='upper right')

    sns.lineplot(x = 'crowding_type', y = 'value', hue = 'sj', 
                data = pd.melt(df_crwd_results[["sj", "crowding_type", "accuracy"]], ['crowding_type', 'sj']), ax = ax4)
    ax4.set_ylabel('Accuracy')
    ax4.set_xlabel('')
    ax4.set_title('Accuracy crowding (without flankers)')
    ax4.set_ylim(0,1.05)
    ax4.legend(fontsize=8, loc='lower right')

    fig.savefig(op.join(plot_path[tsk], 'sj-all_ses-{ses}_task-{task}_mean_RT_ACC_flankered.png'.format(ses = ses_type, task = tsk)))

    ####### PLOT CRITICAL SPACING ########
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,5), dpi=100, facecolor='w', edgecolor='k')

    pt.RainCloud(data = df_crwd_results[df_crwd_results['crowding_type'].isin(crwd_type)], 
                x = 'crowding_type', y = 'critical_spacing', pointplot = True, 
                linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
    ax1.set_ylabel('CS')
    ax1.set_xlabel('')
    ax1.set_title('Critical spacing')
    ax1.set_ylim(0.15,.7)
        
    sns.lineplot(x = 'crowding_type', y = 'value', hue='sj', 
                data = pd.melt(df_crwd_results[df_crwd_results['crowding_type'].isin(crwd_type)][["sj", 
                                                                                                "crowding_type", 
                                                                                                "critical_spacing"]], 
            ['crowding_type', 'sj']), ax = ax2)  
    ax2.set_ylabel('CS')
    ax2.set_xlabel('')
    ax2.set_title('Critical spacing')
    ax2.set_ylim(0.15,.7)
    ax2.legend(fontsize=8, loc='upper right')

    fig.savefig(op.join(plot_path[tsk], 'sj-all_ses-{ses}_task-{task}_CS_distribution.png'.format(ses = ses_type, task = tsk)))


