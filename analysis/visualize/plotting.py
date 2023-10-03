
from tty import CC
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils

import scipy

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter
import matplotlib.ticker as plticker


class PlotsBehavior:
    
    
    def __init__(self, BehObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        BehObj : BehResponses object
            object from one of the classes defined in behaviour.manual_responses
            
        """
        
        # set results object to use later on
        self.BehObj = BehObj
        # if output dir not defined, then make it in derivates
        if outputdir is None:
            self.outputdir = op.join(self.BehObj.dataObj.derivatives_pth,'plots')
        else:
            self.outputdir = outputdir
            
        # number of participants to plot
        self.nr_pp = len(self.BehObj.dataObj.sj_num)

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'
        
    
    def plot_critical_spacing(self, df_CS, save_fig = True):
        
        """ plot critical spacing for group
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
    
        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            fig, ax1 = plt.subplots(1,1, figsize=(4,4), dpi=1000, facecolor='w', edgecolor='k')

            sns.swarmplot(data = df_CS, x = 'crowding_type', y = 'critical_spacing',
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        marker = 'o', size=4, ax = ax1, alpha = .45)

            ax1.plot(df_CS.crowding_type, df_CS.critical_spacing, color='gray', zorder = -1, 
                    alpha = .25, lw=.6)

            sns.pointplot(data = df_CS, x = 'crowding_type', y = 'critical_spacing',
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        size=7, ax = ax1, alpha = 1, n_boot=5000)

            ax1.set_ylabel('CS', fontsize = 16, labelpad = 15)
            ax1.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
            ax1.set_ylim(0.15, .85)
            ax1.tick_params(axis='both', labelsize=14)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_CS.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')
                
    def plot_delta_acc_CS(self, acc_diff_df = None, df_CS = None, save_fig = True):
        
        """ plot accuracy diff for different features + critical spacing for group
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        df_mean2plot = pd.DataFrame({'crowding_type': acc_diff_df.groupby(['crowding_type']).mean().reset_index().crowding_type.values.astype(str),
                                    'acc_diff_color': acc_diff_df.groupby(['crowding_type']).mean().reset_index().acc_diff_color.values.astype(float),
                                    'acc_diff_ori': acc_diff_df.groupby(['crowding_type']).mean().reset_index().acc_diff_ori.values.astype(float)})
    
        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            fig, ax1 = plt.subplots(1,2, figsize=(10, 4), dpi=1000, facecolor='w', edgecolor='k', constrained_layout = True)

            g = sns.scatterplot(data = df_mean2plot, 
                        y = "acc_diff_color", x="acc_diff_ori", hue="crowding_type",
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        ax = ax1[0],s=50)
            ax1[0].errorbar(df_mean2plot.acc_diff_ori.values, 
                        df_mean2plot.acc_diff_color.values, 
                        yerr = acc_diff_df.groupby(['crowding_type']).sem().reset_index().acc_diff_color.values, 
                        xerr = acc_diff_df.groupby(['crowding_type']).sem().reset_index().acc_diff_ori.values,
                        zorder = 0, c='grey', alpha=.9, fmt='none')
            ax1[0].set_xlabel('$\Delta$ Orientation Accuracy (%)', fontsize = 16, labelpad = 15)
            ax1[0].set_ylabel('$\Delta$ Color Accuracy (%)', fontsize = 16, labelpad = 15)
            ax1[0].set_xlim(- 15, 0) #ax1[0].set_xlim(0.84, 1.0)
            ax1[0].set_ylim(- 15, 0) #ax1[0].set_ylim(0.84, 1.0)
            ax1[0].tick_params(axis='both', labelsize=14)
            ax1[0].legend(loc = 'lower right',fontsize=8, title = 'Flanker type')
            # Draw a line of x=y 
            x0, x1 = g.get_xlim()
            y0, y1 = g.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            g.plot(lims, lims, '--k',zorder = -1, alpha = .3)

            loc = plticker.MultipleLocator(base= 5)#.05) # this locator puts ticks at regular intervals
            ax1[0].xaxis.set_major_locator(loc)
            ax1[0].yaxis.set_major_locator(loc)

            g2 = sns.swarmplot(data = df_CS, x = 'crowding_type', y = 'critical_spacing',
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        marker = 'o', size=4, ax = ax1[1], alpha = .45)

            ax1[1].plot(df_CS.crowding_type, df_CS.critical_spacing, color='gray', zorder = -1, 
                    alpha = .25, lw=.6)

            g3 = sns.pointplot(data = df_CS, x = 'crowding_type', y = 'critical_spacing',
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        size=7, ax = ax1[1], alpha = 1, n_boot=5000)

            ax1[1].set_ylabel('CS', fontsize = 16, labelpad = 15)
            ax1[1].set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
            ax1[1].set_ylim(0.15, .85)
            ax1[1].tick_params(axis='both', labelsize=14)

            fig.subplots_adjust(wspace=0.4)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_DeltaFeatureAccuracy_CS.svg'.format(nr = self.nr_pp,
                                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')

    def plot_staircases(self, save_fig = True):
        
        """ plot staircases of each crowding type, per participant
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        
        # loop over subjects
        for i, pp in enumerate(self.BehObj.dataObj.sj_num):
        
            fig, axis = plt.subplots(1, figsize=(10,5), dpi=100)

            sns.lineplot(data = self.BehObj.staircases[self.BehObj.staircases['sj'] == 'sub-{sj}'.format(sj = pp)], 
                         drawstyle='steps-pre', palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'])
            plt.xlabel('# Trials', fontsize=10)
            plt.ylabel('Distance ratio', fontsize=10)
            plt.title('Staircases sub-{sj}'.format(sj = pp), fontsize=10)
            plt.legend(loc = 'upper right',fontsize=7)
            plt.xlim([0, self.BehObj.dataObj.nr_trials_flank])
            plt.ylim(self.BehObj.dataObj.params['crowding']['staircase']['distance_ratio_bounds'][0] - .05,
                    self.BehObj.dataObj.params['crowding']['staircase']['distance_ratio_bounds'][-1] + .05)
            
            if save_fig:
                pp_folder = op.join(self.outputdir, 'sub-{sj}'.format(sj = pp))
                os.makedirs(pp_folder, exist_ok=True)

                fig.savefig(op.join(pp_folder, 'sub-{sj}_ses-{ses}_task-{task}_staircases.png'.format(sj = pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))

    def plot_RT_acc_crowding(self, df_manual_responses = None, df_NoFlanker_results = None, df_mean_results = None,
                                    no_flank = False, save_fig = True):
        
        """ plot RTs and accuracy of crowding task, for group AND each participant
        
        Parameters
        ----------
        no_flank: bool
            if we want to look at crowded or not crowded trials
        save_fig : bool
            save figure in output dir
            
        """
        
        
        # if no flankers 
        if no_flank:
            
            # loop over subjects
            for pp in self.BehObj.dataObj.sj_num:
            
                fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

                pt.RainCloud(data = df_NoFlanker_results[df_NoFlanker_results['sj'] == 'sub-{sj}'.format(sj = pp)], 
                             x = 'type', y = 'mean_RT', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['target_feature_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
                ax1.set_ylabel('RT (seconds)')
                ax1.set_xlabel('')
                ax1.set_title('Mean Reaction Times (without flankers) sub-{sj}'.format(sj = pp))
                ax1.set_ylim(0.2,2)

                pt.RainCloud(data = df_NoFlanker_results[df_NoFlanker_results['sj'] == 'sub-{sj}'.format(sj = pp)], 
                             x = 'type', y = 'accuracy', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['target_feature_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('')
                ax2.set_title('Accuracy (without flankers) sub-{sj}'.format(sj = pp))
                ax2.set_ylim(0,1.05)
                
                if save_fig:
                    pp_folder = op.join(self.outputdir, 'sub-{sj}'.format(sj = pp))
                    os.makedirs(pp_folder, exist_ok=True)

                    fig.savefig(op.join(pp_folder, 'sub-{sj}_ses-{ses}_task-{task}_RT_ACC_UNflankered.png'.format(sj = pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))
                
            # if we have more than 1 participant data in object
            if self.nr_pp > 1: # make group plot

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10), dpi=100, facecolor='w', edgecolor='k')

                pt.RainCloud(data = df_NoFlanker_results, 
                             x = 'type', y = 'mean_RT', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['target_feature_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
                ax1.set_ylabel('RT (seconds)')
                ax1.set_xlabel('')
                ax1.set_title('Reaction Times (without flankers) N = {nr}'.format(nr = self.nr_pp))
                ax1.set_ylim(0.2, 2)

                pt.RainCloud(data = df_NoFlanker_results, 
                             x = 'type', y = 'accuracy', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['target_feature_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('')
                ax2.set_title('Accuracy (without flankers) N = {nr}'.format(nr = self.nr_pp))
                ax2.set_ylim(0, 1.05)
                
                sns.pointplot(x = 'type', y = 'mean_RT', hue = 'sj',
                            data = df_NoFlanker_results, 
                             ax = ax3)
                ax3.set_ylabel('RT (seconds)')
                ax3.set_xlabel('')
                ax3.set_title('Reaction Times (without flankers) N = {nr}'.format(nr = self.nr_pp))
                ax3.set_ylim(0.2, 2)
                ax3.legend([]) #ax3.legend(fontsize=8, loc='upper right')

                sns.pointplot(x = 'type', y = 'accuracy', hue = 'sj', 
                            data = df_NoFlanker_results, 
                              ax = ax4)
                ax4.set_ylabel('Accuracy')
                ax4.set_xlabel('')
                ax4.set_title('Accuracy (without flankers) N = {nr}'.format(nr = self.nr_pp))
                ax4.set_ylim(0, 1.05)
                ax4.legend([]) #ax4.legend(fontsize=8, loc='lower right')
                
                if save_fig:
                    fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT_ACC_UNflankered.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))
        else: # flanked trials
            
            # loop over subjects
            for pp in self.BehObj.dataObj.sj_num:

                fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,4), dpi=100, facecolor='w', edgecolor='k')

                # Reaction Time distribution 
                pt.RainCloud(data = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                                        (df_manual_responses['correct_response'] == 1)], 
                            x = 'crowding_type', y = 'RT', pointplot = True, linewidth = 1,
                            palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
                ax1.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
                ax1.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
                ax1.set_title('Reaction Times Crowding sub-{sj}'.format(sj = pp))
                ax1.set_ylim(0.2,2)

                # Accuracy (actual, and also show accuracy for target color and orientation separately)
                sns.pointplot(data = df_mean_results[df_mean_results['sj'] == 'sub-{sj}'.format(sj = pp)], 
                             x = 'crowding_type', y = 'accuracy_color', 
                             color = self.BehObj.dataObj.params['plotting']['target_feature_colors']['target_color'],
                             label = 'target color', ax = ax2)
                sns.pointplot(data = df_mean_results[df_mean_results['sj'] == 'sub-{sj}'.format(sj = pp)], 
                             x = 'crowding_type', y = 'accuracy_ori',
                             color = self.BehObj.dataObj.params['plotting']['target_feature_colors']['target_ori'],
                             label = 'target orientation', ax = ax2)
                sns.pointplot(data = df_mean_results[df_mean_results['sj'] == 'sub-{sj}'.format(sj = pp)], 
                             x = 'crowding_type', y = 'accuracy', 
                             color = self.BehObj.dataObj.params['plotting']['target_feature_colors']['target_both'],
                             label = 'target both', ax = ax2)
                ax2.set_ylabel('Accuracy', fontsize = 16, labelpad = 15)
                ax2.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
                ax2.set_title('Accuracy Crowding sub-{sj}'.format(sj = pp))
                ax2.set_ylim(0.25, 1.01)
                # quick fix for legend
                handleA = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['target_feature_colors']['target_color'], 
                                         label='target color')
                handleB = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['target_feature_colors']['target_ori'], 
                                         label='target orientation')
                handleC = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['target_feature_colors']['target_both'], 
                                         label='target both')
                ax2.legend(fontsize=8, loc='lower right', 
                           handles = [handleA, handleB, handleC])
                
                if save_fig:
                    pp_folder = op.join(self.outputdir, 'sub-{sj}'.format(sj = pp))
                    os.makedirs(pp_folder, exist_ok=True)

                    fig.savefig(op.join(pp_folder, 'sub-{sj}_ses-{ses}_task-{task}_RT_ACC_flankered.png'.format(sj = pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))
                
            # if we have more than 1 participant data in object
            if self.nr_pp > 1: # make group plot

                ## RT FIGURE
                fig, ax1 = plt.subplots(1,1, figsize=(7,4), dpi=1000, facecolor='w', edgecolor='k')

                pt.RainCloud(data = df_mean_results, 
                            x = 'crowding_type', y = 'mean_RT', pointplot = False, 
                            palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                            linewidth = 1, alpha = .75, dodge = True, saturation = 1, 
                            point_size = 2, width_box = .06, offset = 0.05, move = .1, jitter = 0.05,
                            ax = ax1)
                ax1.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
                ax1.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
                #ax1.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
                ax1.set_ylim(0.2, 2)
                ax1.tick_params(axis='both', labelsize=14)

                if save_fig:
                    fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT_flankered.svg'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)),
                                bbox_inches='tight')
                    
                ## ACCURACY FIGURE
                fig, ax1 = plt.subplots(1,1, figsize=(7,4), dpi=1000, facecolor='w', edgecolor='k')

                pt.RainCloud(data = df_mean_results, 
                            x = 'crowding_type', y = 'accuracy', pointplot = False, 
                            palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                            linewidth = 1, alpha = .75, dodge = True, saturation = 1, 
                            point_size = 2, width_box = .06, offset = 0.05, move = .1, jitter = 0.05,
                            ax = ax1)
                ax1.set_ylabel('Accuracy', fontsize = 16, labelpad = 15)
                ax1.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
                #ax1.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
                ax1.set_ylim(0.25, 1.01)
                ax1.tick_params(axis='both', labelsize=14)

                if save_fig:
                    fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_ACC_flankered.svg'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)),
                                bbox_inches='tight')

                ## extra stuff 

                fig, (ax3, ax4) = plt.subplots(1,2, figsize=(15,4), dpi=100, facecolor='w', edgecolor='k')
                
                sns.pointplot(x = 'crowding_type', y = 'RT', hue = 'sj',
                            data = df_manual_responses[(df_manual_responses['correct_response'] == 1)], 
                             ax = ax3)
                ax3.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
                ax3.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
                ax3.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
                ax3.set_ylim(0.2, 2)
                ax3.legend([]) #ax3.legend(fontsize=8, loc='upper right')

                sns.pointplot(x = 'crowding_type', y = 'accuracy', hue = 'sj', 
                            data = df_mean_results, 
                            ax = ax4)
                ax4.set_ylabel('Accuracy', fontsize = 16, labelpad = 15)
                ax4.set_xlabel('Flanker type', fontsize = 16, labelpad = 15)
                ax4.set_title('Accuracy Crowding N = {nr}'.format(nr = self.nr_pp))
                ax4.set_ylim(0.25, 1.01)
                ax4.legend([]) #ax4.legend(fontsize=8, loc='lower right')
                
                if save_fig:
                    fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT_ACC_singlesj_flankered.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))

    def plot_acc_features_crowding(self, df_mean_results = None, save_fig = True):
        
        """ plot feature accuracy of crowding task (assimilation errors?), for group
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            df_mean2plot = pd.DataFrame({'crowding_type': df_mean_results.groupby(['crowding_type']).mean().reset_index().crowding_type.values.astype(str),
                                        'accuracy_color': df_mean_results.groupby(['crowding_type']).mean().reset_index().accuracy_color.values.astype(float),
                                        'accuracy_ori': df_mean_results.groupby(['crowding_type']).mean().reset_index().accuracy_ori.values.astype(float)})

            fig, ax1 = plt.subplots(1,1, figsize=(4, 4), dpi=1000, facecolor='w', edgecolor='k')

            g = sns.scatterplot(data = df_mean2plot, 
                        y = "accuracy_color", x="accuracy_ori", hue="crowding_type",
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        ax = ax1,s=50)
            ax1.errorbar(df_mean2plot.accuracy_ori.values, 
                        df_mean2plot.accuracy_color.values, 
                        yerr = df_mean_results.groupby(['crowding_type']).sem().reset_index().accuracy_color.values, 
                        xerr = df_mean_results.groupby(['crowding_type']).sem().reset_index().accuracy_ori.values,
                        zorder = 0, c='grey', alpha=.9, fmt='none')
            ax1.set_xlabel('Orientation Percent correct', fontsize = 16, labelpad = 15)
            ax1.set_ylabel('Color Percent correct', fontsize = 16, labelpad = 15)
            ax1.set_xlim(.75, 1.0)
            ax1.set_ylim(.75, 1.0)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.legend(loc = 'lower right',fontsize=8, title = 'Flanker type')

            # Draw a line of x=y 
            x0, x1 = g.get_xlim()
            y0, y1 = g.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            g.plot(lims, lims, '--k',zorder = -1, alpha = .3)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_FeatureAccuracy.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')

    def plot_RT_acc_search(self, df_manual_responses = None, df_mean_results = None, df_search_slopes = None,
                        save_fig = True):
        
        """ plot search reaction times and accuracy
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        
        # loop over subjects
        for i, pp in enumerate(self.BehObj.dataObj.sj_num):
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7), dpi=100, facecolor='w', edgecolor='k')

            #### Reaction Time distribution ####
            pt.RainCloud(data = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                                    (df_manual_responses['correct_response'] == 1)], 
                         x = 'set_size', y = 'RT', pointplot = True, hue='target_ecc',
                         palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax1)
            ax1.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax1.set_ylabel('RT [s]', fontsize = 15, labelpad=15)
            ax1.set_title('Reaction Times Search sub-{sj}'.format(sj = pp), fontsize = 20)
            # set x ticks as integer
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax1.tick_params(axis='both', labelsize = 15)
            ax1.set_ylim(0,6)

            # quick fix for legen
            handleA = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][4],
                                     label = 4)
            handleB = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][8],
                                     label = 8)
            handleC = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][12],
                                     label = 12)
            ax1.legend(loc = 'upper left',fontsize=12, handles = [handleA, handleB, handleC], 
                       title="Target ecc", fancybox=True)

            #### Accuracy ####
            sns.pointplot(data = df_mean_results[(df_mean_results['sj'] == 'sub-{sj}'.format(sj = pp))],
                          x = 'set_size', y = 'accuracy', hue='target_ecc',
                         palette = self.BehObj.dataObj.params['plotting']['ecc_colors'], ax = ax2)
            ax2.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax2.set_ylabel('Accuracy', fontsize = 15, labelpad=15)
            ax2.set_title('Accuracy Search sub-{sj}'.format(sj = pp), fontsize = 20)
            # set x ticks as integer
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax2.tick_params(axis='both', labelsize = 15)
            ax2.set_ylim(0.5,1)

            # quick fix for legen
            ax2.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                       title="Target ecc", fancybox=True)
            
            if save_fig:
                pp_folder = op.join(self.outputdir, 'sub-{sj}'.format(sj = pp))
                os.makedirs(pp_folder, exist_ok=True)

                fig.savefig(op.join(pp_folder, 'sub-{sj}_ses-{ses}_task-{task}_VSearch_RT_acc.png'.format(sj = pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))
                
        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(30,20), dpi=100, facecolor='w', edgecolor='k')

            #### Search Reaction times, as a function of set size and ecc ####
            pt.RainCloud(data = df_mean_results,
                         x = 'set_size', y = 'mean_RT', pointplot = True, hue='target_ecc',
                         palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax1)
            ax1.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax1.set_ylabel('RT [s]', fontsize = 15, labelpad=15)
            ax1.set_title('Mean Reaction Times Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax1.tick_params(axis='both', labelsize = 15)
            ax1.set_ylim(0.5,5.5)
            ax1.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                       title="Target ecc", fancybox=True)

            #### Search accuracy, as a function of set size and ecc ####
            pt.RainCloud(data = df_mean_results,
                         x = 'set_size', y = 'accuracy', pointplot = True, hue='target_ecc',
                         palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax2)
            ax2.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax2.set_ylabel('Accuracy', fontsize = 15, labelpad=15)
            ax2.set_title('Accuracy Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax2.tick_params(axis='both', labelsize = 15)
            ax2.set_ylim(0.5,1)

            # quick fix for legend
            ax2.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                       title="Target ecc", fancybox=True)

            #### Search Reaction times, per participant ####
            sns.pointplot(x = 'set_size', y = 'mean_RT', hue = 'sj',
                        data = df_mean_results, 
                         ax = ax3)
            ax3.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax3.set_ylabel('RT [s]', fontsize = 15, labelpad=15)
            ax3.set_title('Mean Reaction Times Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax3.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax3.tick_params(axis='both', labelsize = 15)
            ax3.set_ylim(0.5,5.5)
            ax3.legend([]) #ax3.legend(loc = 'lower right',fontsize=10, fancybox=True)

            #### Search Slopes, as a function of ecc ####
            pt.RainCloud(data = df_search_slopes,
                         x = 'target_ecc', y = 'slope', pointplot = True, #hue='target_ecc',
                         palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax4)
            ax4.set_xlabel('Target Eccentricity', fontsize = 15, labelpad=15)
            ax4.set_ylabel('RT/set size [ms/item]', fontsize = 15, labelpad=15)
            ax4.set_title('Search Slopes N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax4.tick_params(axis='both', labelsize = 15)
            ax4.set_ylim(0,110)
            
            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_VSearch_RT_acc.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))
                
            ### plot main RT group figure ###
            
            # make fake dataframe, to fill with nans
            # so we force plot to be in continuous x-axis
            fake_ss = np.array([i for i in np.arange(int(self.BehObj.dataObj.set_size[-1])) if i not in self.BehObj.dataObj.set_size[:2]])

            tmp_df = pd.DataFrame({'sj': [], 'target_ecc': [], 'set_size': [], 'mean_RT': [], 'accuracy': []})
            for s in df_mean_results.sj.unique():
                
                tmp_df = pd.concat((tmp_df,
                                pd.DataFrame({'sj': np.repeat(s,len(np.repeat(fake_ss,3))), 
                                                'target_ecc': np.tile(self.BehObj.dataObj.num_ecc,len(fake_ss)), 
                                                'set_size': np.repeat(fake_ss,3), 
                                                'mean_RT': np.repeat(np.nan,len(np.repeat(fake_ss,3))), 
                                                'accuracy': np.repeat(np.nan,len(np.repeat(fake_ss,3)))})))
            fake_DF = pd.concat((df_mean_results,tmp_df))

            ## actually plot
            fig, ax1 = plt.subplots(1,1, figsize=(9,3), dpi=1000, facecolor='w', edgecolor='k')

            pt.RainCloud(data = fake_DF, #df_mean_results, 
                        x = 'set_size', y = 'mean_RT', pointplot = False, hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linewidth = 1,
                        alpha = .9, dodge = True, saturation = 1, 
                        point_size = 1.8, width_viol = 3,
                        width_box = .72, offset = 0.45, move = .8, jitter = 0.25,
                        ax = ax1)
            ax1.set_xlabel('Set Size [items]', fontsize = 16, labelpad = 15)
            ax1.set_ylabel('RT [s]', fontsize = 16, labelpad = 15)
            #ax1.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
            ax1.set_ylim(0.5,5.1)
            ax1.set_xlim(5,33)

            ax1.tick_params(axis='both', labelsize=14)
            ax1.xaxis.set_ticks(np.arange(7, 33, 3))
            ax1.set_xticklabels([str(i) for i in np.arange(7, 33, 3)])
            #ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(search_behaviour.dataObj.set_size[x]))) 

            # quick fix for legen
            handleA = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][4],
                                    label = '4 deg')
            handleB = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][8],
                                    label = '8 deg')
            handleC = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][12],
                                    label = '12 deg')
            ax1.legend(loc = 'upper left',fontsize=8, handles = [handleA, handleB, handleC], 
                                title="Target ecc")#, fancybox=True)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')

    def plot_correlations_RT_CS(self, df_CS = None, df_mean_results = None, 
                                    crowding_type_list = ['orientation', 'color', 'conjunction'],
                                    save_fig = True, outdir = None):
        
        """ plot correlations between search reaction times and CS
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        
        # plot for each crowding type separately 

        for crowding_type in crowding_type_list:

            # build tidy dataframe with relevant info
            corr_df4plotting = pd.DataFrame([])

            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_mean_results[(df_mean_results['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                    (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                
                # append
                corr_df4plotting = pd.concat((corr_df4plotting,
                                            tmp_df.copy()))

            ## plot ecc x set size grid
            # of correlations

            # facetgrid of plots
            g = sns.lmplot(
                data = corr_df4plotting, x = 'critical_spacing', y = 'mean_RT',markers = 'x',
                col = 'target_ecc', row = 'set_size', height = 4, hue = 'target_ecc',
                palette = [self.BehObj.dataObj.params['plotting']['crwd_type_colors'][crowding_type]],
                facet_kws = dict(sharex = True, sharey = True)
            )

            # main axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type.capitalize()), 'RT [s]', fontsize = 18, labelpad=15)
            g.set(xlim=(.15, .71), ylim=(0, 5))

            ## set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
                ax.set_title(title, fontsize = 22, pad = 30)
                ax.tick_params(axis='both', labelsize=14)

            # remove unecessary title
            for rax in g.axes[1:]:
                for cax in rax:
                    cax.set_title('')
                    cax.tick_params(axis='both', labelsize=14)

            # add row title
            for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.BehObj.dataObj.set_size))]): # last column
                ax.text(.8, 2.5, '{s} items'.format(s = self.BehObj.dataObj.set_size[ind]) , rotation = 0, fontsize = 22)

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
                for ss_ind, ss in enumerate(self.BehObj.dataObj.set_size):
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].mean_RT.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                    g.axes[e_ind, ss_ind].text(.48, 4.5, 
                                            r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                                        '\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                                            horizontalalignment='left', fontsize = 16, weight='bold')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_RT_correlation_CS_{ct}.svg'.format(nr = self.nr_pp,
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)),
                        bbox_inches='tight')

        # ############ also correlate with the mean CS (mean over crowding types) #############
        # mean_df_CS = df_CS.groupby('sj').mean().reset_index()

        # # build tidy dataframe with relevant info
        # corr_df4plotting = pd.DataFrame([])

        # # loop over subjects
        # for _, pp in enumerate(self.BehObj.dataObj.sj_num):

        #     # make temporary dataframe
        #     tmp_df = df_mean_results[(df_mean_results['sj']== 'sub-{s}'.format(s = pp))]
        #     tmp_df['critical_spacing'] = mean_df_CS[(mean_df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
            
        #     # append
        #     corr_df4plotting = pd.concat((corr_df4plotting,
        #                                 tmp_df.copy()))

        # ## plot ecc x set size grid
        # # of correlations

        # # facetgrid of plots
        # g = sns.lmplot(
        #     data = corr_df4plotting, x = 'critical_spacing', y = 'mean_RT',
        #     col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
        #     facet_kws = dict(sharex = True, sharey = True)
        # )

        # # main axis labels
        # g.set_axis_labels('Mean CS', 'RT (s)', fontsize = 10, labelpad=15)
        # g.set(xlim=(.15, .75), ylim=(0, 5))

        # ## set subplot titles
        # for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
        #     ax.set_title(title, fontsize = 15, pad = 25)
            
        # # remove unecessary title
        # for rax in g.axes[1:]:
        #     for cax in rax:
        #         cax.set_title('')
                
        # # add row title
        # for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.BehObj.dataObj.set_size))]): # last column
        #     ax.text(.8, 2.5, '{s} items'.format(s = self.BehObj.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

        # ## add Spearman correlation value and p-val 
        # # as annotation
        # for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
        #     for ss_ind, ss in enumerate(self.BehObj.dataObj.set_size):
        #         rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
        #                         (corr_df4plotting['set_size'] == ss)].mean_RT.values, 
        #                             corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
        #                         (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

        #         g.axes[e_ind, ss_ind].text(.2, 4.5, 
        #                                 'rho = %.2f \np-value = %.3f'%(rho,pval), 
        #                                 horizontalalignment='left')

        # if save_fig:
        #     g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_SearchRT_CS-mean.png'.format(nr = self.nr_pp,
        #                                                                                             ses = self.BehObj.dataObj.session)))

    def plot_correlations_slopeRT_CS(self, df_CS = None, df_search_slopes = None, 
                                        crowding_type_list = ['orientation', 'color', 'conjunction'],
                                        save_fig = True, outdir = None, seed_num = 42):
        
        """ plot correlations between search reaction times SLOPEs and CS
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        # build tidy dataframe with relevant info
        corr_slope_df4plotting = pd.DataFrame([])

        for crowding_type in crowding_type_list:
            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_search_slopes[(df_search_slopes['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                            (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                tmp_df['crowding_type'] = np.tile(crowding_type, len(tmp_df))

                # append
                corr_slope_df4plotting = pd.concat((corr_slope_df4plotting,
                                            tmp_df.copy()))
                
        ## plot correlations per crowding type 
        g = sns.lmplot(
            data = corr_slope_df4plotting, x = 'critical_spacing', y = 'slope',
            col= 'crowding_type', height = 4, markers = 'x',
            hue = 'crowding_type', palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
            facet_kws = dict(sharex = True, sharey = True)
        )

        # axis labels
        g.set_axis_labels('CS', 'RT/set size (ms/item)', fontsize = 18, labelpad=15, fontweight="bold")
        g.set(xlim=(.15, .71), ylim=(0, 120))

        ## set subplot titles
        for ax, title in zip(g.axes[0], ['{ct}'.format(ct = c).capitalize() for c in df_CS.crowding_type.unique()]):
            ax.set_title(title, fontsize = 22, pad = 30)
            ax.tick_params(axis='both', labelsize=14)

        ## add Spearman correlation value and p-val 
        # as annotation
        p_val_all = []
        for ct_ind, ct in enumerate(df_CS.crowding_type.unique()):
            
            rho, pval = scipy.stats.spearmanr(corr_slope_df4plotting[(corr_slope_df4plotting['crowding_type'] == ct)].slope.values, 
                                        corr_slope_df4plotting[(corr_slope_df4plotting['crowding_type'] == ct)].critical_spacing.values)
            p_val_all.append(pval)

            g.axes[0][ct_ind].text(.65, 15, #.2, 105, 
                        r"$\rho$ = {r}".format(r = '%.2f'%(rho)), #+\
                        #'\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                        horizontalalignment='right', fontsize = 16, weight='bold')

        if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_RT_slope_correlation_CS.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session)),
                        bbox_inches='tight')
        
        ## permutate correlations and plot
        fig, ax1 = plt.subplots(1,3, figsize=(13, 5), dpi=100, facecolor='w', edgecolor='k', sharey = True)

        for ct_ind, ct in enumerate(df_CS.crowding_type.unique()):
            
            slope_arr = corr_slope_df4plotting[(corr_slope_df4plotting['crowding_type'] == ct)].slope.values
            cs_arr = corr_slope_df4plotting[(corr_slope_df4plotting['crowding_type'] == ct)].critical_spacing.values
            
            # get observed correlation value
            rho, pval = scipy.stats.spearmanr(slope_arr, cs_arr)
            
            # get permutation values
            perm_rho, pval_perm = utils.permutation_correlations(slope_arr, cs_arr, method = 'spearman',
                                                                perm_num=10000, seed = seed_num + ct_ind,
                                                                p_val_side='two-sided')
            
            ax1[ct_ind].hist(perm_rho, color = self.BehObj.dataObj.params['plotting']['crwd_type_colors'][ct], 
                            edgecolor='k', alpha=0.65,bins=50)
            ax1[ct_ind].axvline(rho, color='black', linestyle='dashed', linewidth=1.5)
            ax1[ct_ind].text(-.45, 550, 
                            #'observed\n'+\
                        # r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                            '\n\npermutation'+\
                        '\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval_perm))[1:],
                        horizontalalignment='left', fontsize = 14, weight='bold')
            
            ax1[ct_ind].set_xlabel('Permutation '+r"$\rho$", fontsize=18, labelpad = 15)
            
        # axis labels
        ax1[0].set_ylabel('Frequency', fontsize=18, labelpad = 15)

        ## set subplot titles
        for ax, title in zip(ax1, ['{ct}'.format(ct = c).capitalize() for c in df_CS.crowding_type.unique()]):
            ax.set_title(title, fontsize = 22, pad = 30)
            ax.tick_params(axis='both', labelsize=14)
            ax.set(xlim=(-.5, .5))
        
        if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_RT_slope_correlation_CS_permutation.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session)),
                        bbox_inches='tight')

    def plot_RTreliability(self, rho_sh_RT = None, save_fig = True, df_search_slopes_p1 = None, df_search_slopes_p2 = None):

        """
        Plot histogram with split half reliabilities
        and if provided, example of split half
        """

        g = sns.histplot(rho_sh_RT, bins = 25)
        #g.set(xlabel=)
        g.tick_params(axis='both', labelsize=15)
        g.set_xlabel(r'Split-half reliability $\rho$',fontsize=15, labelpad = 15)
        g.set_ylabel('Count', fontsize=15)
        #g.set_xlim([.75, 1])

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_VSearch_SplitHalf_Histogram.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.BehObj.dataObj.session, 
                                                                                                        task = self.BehObj.dataObj.task_name)))
            
        if df_search_slopes_p1 is not None and df_search_slopes_p2 is not None:
            
            # build tidy dataframe with relevant info
            corr_slope_df4plotting = pd.DataFrame([])

            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

                # append
                corr_slope_df4plotting = pd.concat((corr_slope_df4plotting,
                                            pd.DataFrame({'sj': ['sub-{s}'.format(s = pp)], 
                                                        'P1_slope': df_search_slopes_p1[df_search_slopes_p1['sj']== 'sub-{s}'.format(s = pp)].slope.values, 
                                                        'P2_slope': df_search_slopes_p2[df_search_slopes_p2['sj']== 'sub-{s}'.format(s = pp)].slope.values})))
                    
            ## plot 
            g = sns.lmplot(
                data = corr_slope_df4plotting, x = 'P1_slope', y = 'P2_slope',
                height = 4, markers = 'x')

            # axis labels
            g.set_axis_labels('Half_1\n\nRT/set size (ms/item)', 'Half_2\n\nRT/set size (ms/item)', fontsize = 18, labelpad=15, fontweight="bold")
            g.set(xlim=(0, 120), ylim=(0, 120))

            ## add Spearman correlation value and p-val 
            # as annotation
            rho, pval = scipy.stats.spearmanr(df_search_slopes_p1.slope.values, df_search_slopes_p2.slope.values)

            g.axes[0][0].text(100, 15, #.2, 105, 
                        r"$\rho$ = {r}".format(r = '%.2f'%(rho)) +\
                        '\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                        horizontalalignment='right', fontsize = 16, weight='bold')
            
            if save_fig:
                g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_VSearch_SplitHalf_Regression.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))


    def plot_CS_types_correlation(self, df_CS = None, save_fig = True):

        """ plot correlations between different CS types
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        # organize CS values in tidy DF
        corr_df4plotting = self.BehObj.combine_CS_df(df_CS)

        ### COLOR vs ORIENTATION CS ###
        g = sns.jointplot(data = corr_df4plotting, x = 'color', y = 'orientation', kind = 'reg', marker = 'x',
                        xlim = (.15,.71), ylim = (.15,.71), marginal_ticks = True, color = 'grey',#'#918e91',#'#baafba',
                        marginal_kws=dict(bins=10, fill=True, kde=False, stat='percent'),
                        height=6, ratio=3, space = .5)

        # Draw a line of x=y 
        x0, x1 = g.ax_joint.get_xlim()
        y0, y1 = g.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.ax_joint.plot(lims, lims, '--k',zorder = -1, alpha = .3)

        # margin axis labels
        #g.ax_marg_x.set_xlabel('Percent')#, fontsize = 12, labelpad = 15)
        g.ax_marg_x.tick_params(axis='both', labelsize=18)
        g.ax_marg_y.tick_params(axis='both', labelsize=18)

        # add color to histogram
        plt.setp(g.ax_marg_y.patches, 
                color = self.BehObj.dataObj.params['plotting']['crwd_type_colors']['orientation'], alpha=.75)
        plt.setp(g.ax_marg_x.patches, 
                color = self.BehObj.dataObj.params['plotting']['crwd_type_colors']['color'], alpha=.75)

        # add annotation with spearman correlation
        rho, pval = scipy.stats.spearmanr(corr_df4plotting.color.values, 
                                                corr_df4plotting.orientation.values)
        g.ax_joint.text(.58, .18, 
                        r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                        '\n{p}'.format(p = '$\it{p}$ < .001'), #+r"= {:.2e}".format(pval)),
                        horizontalalignment='left', fontsize = 18, fontweight='bold')
        # axis labels
        g.ax_joint.set_ylabel('Orientation CS', fontsize = 22, labelpad = 15)
        g.ax_joint.set_xlabel('Color CS', fontsize = 22, labelpad = 15)
        g.ax_joint.tick_params(axis='both', labelsize=18)
        # set the ticks first
        g.ax_joint.set_xticks(np.around(np.linspace(.15,.71,7),decimals=1))
        g.ax_joint.set_yticks(np.around(np.linspace(.15,.71,7),decimals=1))

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_CS_correlation_ORI_COLOR.svg'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )),
                    bbox_inches='tight')

        ### COLOR vs CONJUNCTION CS ###
        g = sns.jointplot(data = corr_df4plotting, x = 'color', y = 'conjunction', kind = 'reg', marker = 'x',
                        xlim = (.15,.71), ylim = (.15,.71), marginal_ticks = True, color = 'grey',#'#918e91',#'#baafba',
                        marginal_kws=dict(bins=10, fill=True, kde=False, stat='percent'),
                        height=6, ratio=3, space = .5)

        # Draw a line of x=y 
        x0, x1 = g.ax_joint.get_xlim()
        y0, y1 = g.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.ax_joint.plot(lims, lims, '--k',zorder = -1, alpha = .3)

        # margin axis labels
        #g.ax_marg_x.set_xlabel('Percent')#, fontsize = 12, labelpad = 15)
        g.ax_marg_x.tick_params(axis='both', labelsize=18)
        g.ax_marg_y.tick_params(axis='both', labelsize=18)

        # add color to histogram
        plt.setp(g.ax_marg_y.patches, 
                color = self.BehObj.dataObj.params['plotting']['crwd_type_colors']['conjunction'], alpha=.75)
        plt.setp(g.ax_marg_x.patches, 
                color = self.BehObj.dataObj.params['plotting']['crwd_type_colors']['color'], alpha=.75)

        # add annotation with spearman correlation
        rho, pval = scipy.stats.spearmanr(corr_df4plotting.color.values, 
                                                corr_df4plotting.conjunction.values)
        g.ax_joint.text(.58, .18, 
                        r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                        '\n{p}'.format(p = '$\it{p}$ < .001'), #+r"= {:.2e}".format(pval)),
                        horizontalalignment='left', fontsize = 18, fontweight='bold')
        # axis labels
        g.ax_joint.set_ylabel('Conjunction CS', fontsize = 22, labelpad = 15)
        g.ax_joint.set_xlabel('Color CS', fontsize = 22, labelpad = 15)
        g.ax_joint.tick_params(axis='both', labelsize=18)
        # set the ticks first
        g.ax_joint.set_xticks(np.around(np.linspace(.15,.71,7),decimals=1))
        g.ax_joint.set_yticks(np.around(np.linspace(.15,.71,7),decimals=1))  

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_CS_correlation_CONJ_COLOR.svg'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )),
                    bbox_inches='tight')

        ### ORIENTATION vs CONJUNCTION CS ### 
        g = sns.jointplot(data = corr_df4plotting, x = 'conjunction', y = 'orientation', kind = 'reg', marker = 'x',
                        xlim = (.15,.71), ylim = (.15,.71), marginal_ticks = True, color = 'grey',#'#918e91',#'#baafba',
                        marginal_kws=dict(bins=10, fill=True, kde=False, stat='percent'),
                        height=6, ratio=3, space = .5)

        # Draw a line of x=y 
        x0, x1 = g.ax_joint.get_xlim()
        y0, y1 = g.ax_joint.get_ylim()
        lims = [max(x0, y0), min(x1, y1)]
        g.ax_joint.plot(lims, lims, '--k',zorder = -1, alpha = .3)

        # margin axis labels
        #g.ax_marg_x.set_xlabel('Percent')#, fontsize = 12, labelpad = 15)
        g.ax_marg_x.tick_params(axis='both', labelsize=18)
        g.ax_marg_y.tick_params(axis='both', labelsize=18)

        # add color to histogram
        plt.setp(g.ax_marg_y.patches, 
                color = self.BehObj.dataObj.params['plotting']['crwd_type_colors']['orientation'], alpha=.75)
        plt.setp(g.ax_marg_x.patches, 
                color = self.BehObj.dataObj.params['plotting']['crwd_type_colors']['conjunction'], alpha=.75)

        # add annotation with spearman correlation
        rho, pval = scipy.stats.spearmanr(corr_df4plotting.conjunction.values, 
                                                corr_df4plotting.orientation.values)
        g.ax_joint.text(.58, .18, 
                        r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                        '\n{p}'.format(p = '$\it{p}$ < .001'), #+r"= {:.2e}".format(pval)),
                        horizontalalignment='left', fontsize = 18, fontweight='bold')
        # axis labels
        g.ax_joint.set_ylabel('Orientation CS', fontsize = 22, labelpad = 15)
        g.ax_joint.set_xlabel('Conjunction CS', fontsize = 22, labelpad = 15)
        g.ax_joint.tick_params(axis='both', labelsize=18)
        # set the ticks first
        g.ax_joint.set_xticks(np.around(np.linspace(.15,.71,7),decimals=1))
        g.ax_joint.set_yticks(np.around(np.linspace(.15,.71,7),decimals=1)) 

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_CS_correlation_ORI_CONJ.svg'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )),
                    bbox_inches='tight')

    def plot_correlations_RT_CS_heatmap(self, df_CS = None, df_mean_results = None, method = 'pearson',
                                        crowding_type_list = ['orientation', 'color', 'conjunction'],
                                        save_fig = True, outdir = None):
        
        """ plot correlations between search reaction times and CS
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        
        # plot for each crowding type separately 

        for crowding_type in crowding_type_list:

            # build tidy dataframe with relevant info
            corr_df4plotting = pd.DataFrame([])

            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_mean_results[(df_mean_results['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                    (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                
                # append
                corr_df4plotting = pd.concat((corr_df4plotting,
                                            tmp_df.copy()))

            # get correlation dfs for each case
            # but need to replace column names
            
            corr_df, pval_df = self.BehObj.make_search_CS_corr_2Dmatrix(corr_df4plotting.rename(columns={'mean_RT': 'y_val', 
                                                                                                'critical_spacing': 'x_val'}), 
                                                                    method = method)

            ## make figure
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,3), dpi=100, facecolor='w', edgecolor='k')

            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-.35, vmax=.35, ax=ax1)
            ax1.set_title('{mt} correlation'.format(mt = method), fontsize = 10)

            sns.heatmap(pval_df, annot=True, vmin=0.00, vmax=0.10, cmap='RdGy', ax=ax2)
            ax2.set_title('{mt} p-values'.format(mt = method), fontsize = 10)

            if save_fig:
                fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_{mt}_correlations_SearchRT_CS-{ct}_heatmap.png'.format(nr = self.nr_pp,
                                                                                                        mt = method,
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)))

    def plot_correlations_slopeRT_CS_heatmap(self, df_CS = None, df_search_slopes = None, method = 'pearson',
                                                crowding_type_list = ['orientation', 'color', 'conjunction'],
                                                save_fig = True, outdir = None):
        
        """ plot correlations between search reaction times SLOPEs and CS
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        
        # plot for each crowding type separately 

        for crowding_type in crowding_type_list:

            # build tidy dataframe with relevant info
            corr_slope_df4plotting = pd.DataFrame([])

            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_search_slopes[(df_search_slopes['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                            (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                
                # append
                corr_slope_df4plotting = pd.concat((corr_slope_df4plotting,
                                            tmp_df.copy()))


            # get correlation dfs for each case
            # but need to replace column names
            
            corr_df, pval_df = self.BehObj.make_search_CS_corr_1Dmatrix(corr_slope_df4plotting.rename(columns={'slope': 'y_val', 
                                                                                                            'critical_spacing': 'x_val'}), 
                                                                    method = method)

            ## make figure
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,1), dpi=100, facecolor='w', edgecolor='k')

            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-.35, vmax=.35, ax=ax1, 
                        yticklabels=[])
            ax1.set_title('{mt} correlation'.format(mt = method), fontsize = 10)

            sns.heatmap(pval_df, annot=True, vmin=0.00, vmax=0.10, cmap='RdGy', ax=ax2,
                        yticklabels=[])
            ax2.set_title('{mt} p-values'.format(mt = method), fontsize = 10)

            if save_fig:
                fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_{mt}_correlations_SearchSlopeRT_CS-{ct}_heatmap.png'.format(nr = self.nr_pp,
                                                                                                        mt = method,
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)))