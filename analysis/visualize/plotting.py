
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
        
    
    def plot_critical_spacing(self, df_CS, save_fig = True):
        
        """ plot critical spacing for group
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
    
        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')
            
            pt.RainCloud(data = df_CS, 
                        x = 'crowding_type', y = 'critical_spacing', pointplot = True, 
                        palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
            ax1.set_ylabel('CS')
            ax1.set_xlabel('')
            ax1.set_title('Critical spacing N = {nr}'.format(nr = self.nr_pp))
            ax1.set_ylim(self.BehObj.dataObj.params['crowding']['staircase']['distance_ratio_bounds'][0],
                        self.BehObj.dataObj.params['crowding']['staircase']['distance_ratio_bounds'][-1])

            sns.pointplot(x = 'crowding_type', y = 'critical_spacing', hue = 'sj',
                            data = df_CS, 
                             ax = ax2) 
            ax2.set_ylabel('CS')
            ax2.set_xlabel('')
            ax2.set_title('Critical spacing N = {nr}'.format(nr = self.nr_pp))
            ax1.set_ylim(self.BehObj.dataObj.params['crowding']['staircase']['distance_ratio_bounds'][0],
                        self.BehObj.dataObj.params['crowding']['staircase']['distance_ratio_bounds'][-1])
            ax2.legend(fontsize=8, loc='upper right')
            
            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_CS_distribution.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))

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

                fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=100, facecolor='w', edgecolor='k')

                # Reaction Time distribution 
                pt.RainCloud(data = df_manual_responses[(df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                                        (df_manual_responses['correct_response'] == 1)], 
                             x = 'crowding_type', y = 'RT', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
                ax1.set_ylabel('RT (seconds)')
                ax1.set_xlabel('')
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
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('')
                ax2.set_title('Accuracy Crowding sub-{sj}'.format(sj = pp))
                ax2.set_ylim(0,1.05)
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

                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10), dpi=100, facecolor='w', edgecolor='k')

                pt.RainCloud(data = df_mean_results, 
                             x = 'crowding_type', y = 'mean_RT', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax1)
                ax1.set_ylabel('RT (seconds)')
                ax1.set_xlabel('')
                ax1.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
                ax1.set_ylim(0.2, 2)

                pt.RainCloud(data = df_mean_results, 
                             x = 'crowding_type', y = 'accuracy', pointplot = True, 
                             palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                            linecolor = 'grey',alpha = .75, dodge = True, saturation = 1, ax = ax2)
                ax2.set_ylabel('Accuracy')
                ax2.set_xlabel('')
                ax2.set_title('Accuracy Crowding N = {nr}'.format(nr = self.nr_pp))
                ax2.set_ylim(0, 1.05)
                
                sns.pointplot(x = 'crowding_type', y = 'RT', hue = 'sj',
                            data = df_manual_responses[(df_manual_responses['correct_response'] == 1)], 
                             ax = ax3)
                ax3.set_ylabel('RT (seconds)')
                ax3.set_xlabel('')
                ax3.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
                ax3.set_ylim(0.2, 2)
                ax3.legend([]) #ax3.legend(fontsize=8, loc='upper right')

                sns.pointplot(x = 'crowding_type', y = 'accuracy', hue = 'sj', 
                            data = df_mean_results, 
                              ax = ax4)
                ax4.set_ylabel('Accuracy')
                ax4.set_xlabel('')
                ax4.set_title('Accuracy Crowding N = {nr}'.format(nr = self.nr_pp))
                ax4.set_ylim(0, 1.05)
                ax4.legend([]) #ax4.legend(fontsize=8, loc='lower right')
                
                if save_fig:
                    fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT_ACC_flankered.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))


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
                data = corr_df4plotting, x = 'critical_spacing', y = 'mean_RT',
                col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
                facet_kws = dict(sharex = True, sharey = True)
            )

            # main axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type), 'RT (s)', fontsize = 10, labelpad=15)
            g.set(xlim=(.15, .75), ylim=(0, 5))

            ## set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
                ax.set_title(title, fontsize = 15, pad = 25)
                
            # remove unecessary title
            for rax in g.axes[1:]:
                for cax in rax:
                    cax.set_title('')
                    
            # add row title
            for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.BehObj.dataObj.set_size))]): # last column
                ax.text(.8, 2.5, '{s} items'.format(s = self.BehObj.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
                for ss_ind, ss in enumerate(self.BehObj.dataObj.set_size):
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].mean_RT.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                    g.axes[e_ind, ss_ind].text(.2, 4.5, 
                                            'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                            horizontalalignment='left')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_SearchRT_CS-{ct}.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)))


        ############ also correlate with the mean CS (mean over crowding types) #############
        mean_df_CS = df_CS.groupby('sj').mean().reset_index()

        # build tidy dataframe with relevant info
        corr_df4plotting = pd.DataFrame([])

        # loop over subjects
        for _, pp in enumerate(self.BehObj.dataObj.sj_num):

            # make temporary dataframe
            tmp_df = df_mean_results[(df_mean_results['sj']== 'sub-{s}'.format(s = pp))]
            tmp_df['critical_spacing'] = mean_df_CS[(mean_df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
            
            # append
            corr_df4plotting = pd.concat((corr_df4plotting,
                                        tmp_df.copy()))

        ## plot ecc x set size grid
        # of correlations

        # facetgrid of plots
        g = sns.lmplot(
            data = corr_df4plotting, x = 'critical_spacing', y = 'mean_RT',
            col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
            facet_kws = dict(sharex = True, sharey = True)
        )

        # main axis labels
        g.set_axis_labels('Mean CS', 'RT (s)', fontsize = 10, labelpad=15)
        g.set(xlim=(.15, .75), ylim=(0, 5))

        ## set subplot titles
        for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
            ax.set_title(title, fontsize = 15, pad = 25)
            
        # remove unecessary title
        for rax in g.axes[1:]:
            for cax in rax:
                cax.set_title('')
                
        # add row title
        for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.BehObj.dataObj.set_size))]): # last column
            ax.text(.8, 2.5, '{s} items'.format(s = self.BehObj.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

        ## add Spearman correlation value and p-val 
        # as annotation
        for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
            for ss_ind, ss in enumerate(self.BehObj.dataObj.set_size):
                rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                (corr_df4plotting['set_size'] == ss)].mean_RT.values, 
                                    corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                g.axes[e_ind, ss_ind].text(.2, 4.5, 
                                        'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                        horizontalalignment='left')

        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_SearchRT_CS-mean.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session)))



    def plot_correlations_slopeRT_CS(self, df_CS = None, df_search_slopes = None, 
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

            ## plot correlations per ecc 
            g = sns.lmplot(
                data = corr_slope_df4plotting, x = 'critical_spacing', y = 'slope',
                col = 'target_ecc', height = 3, #palette = 'flare',
                facet_kws = dict(sharex = True, sharey = True)
            )

            # axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type), 'RT/set size (ms/item)', fontsize = 10, labelpad=15)
            g.set(xlim=(.15, .75), ylim=(0, 120))

            # set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
                ax.set_title(title, fontsize = 15, pad = 25)
                

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
                rho, pval = scipy.stats.spearmanr(corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].slope.values, 
                                        corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].critical_spacing.values)

                g.axes[0][e_ind].text(.2, 110, 
                                'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                horizontalalignment='left')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_SearchSlopeRT_CS-{ct}.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)))

        ############ also correlate with the mean CS (mean over crowding types) #############
        mean_df_CS = df_CS.groupby('sj').mean().reset_index()

        # build tidy dataframe with relevant info
        corr_slope_df4plotting = pd.DataFrame([])

        # loop over subjects
        for _, pp in enumerate(self.BehObj.dataObj.sj_num):

            # make temporary dataframe
            tmp_df = df_search_slopes[(df_search_slopes['sj']== 'sub-{s}'.format(s = pp))]
            tmp_df['critical_spacing'] = mean_df_CS[(mean_df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
            
            # append
            corr_slope_df4plotting = pd.concat((corr_slope_df4plotting,
                                        tmp_df.copy()))

        ## plot correlations per ecc 
        g = sns.lmplot(
            data = corr_slope_df4plotting, x = 'critical_spacing', y = 'slope',
            col = 'target_ecc', height = 3, #palette = 'flare',
            facet_kws = dict(sharex = True, sharey = True)
        )

        # axis labels
        g.set_axis_labels('Mean CS', 'RT/set size (ms/item)', fontsize = 10, labelpad=15)
        g.set(xlim=(.15, .75), ylim=(0, 120))

        # set subplot titles
        for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
            ax.set_title(title, fontsize = 15, pad = 25)
            

        ## add Spearman correlation value and p-val 
        # as annotation
        for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
            rho, pval = scipy.stats.spearmanr(corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].slope.values, 
                                    corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].critical_spacing.values)

            g.axes[0][e_ind].text(.2, 110, 
                            'rho = %.2f \np-value = %.3f'%(rho,pval), 
                            horizontalalignment='left')

        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_SearchSlopeRT_CS-mean.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session)))


    def plot_CS_types_correlation(self, df_CS = None, save_fig = True):

        """ plot correlations between different CS types
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        # organize CS values in tidy DF
        corr_df4plotting = self.BehObj.combine_CS_df(df_CS)

        # COLOR vs ORIENTATION CS
        g = sns.lmplot(data = corr_df4plotting, x = 'color', y = 'orientation', height = 5)

        rho, pval = scipy.stats.spearmanr(corr_df4plotting.color.values, 
                                        corr_df4plotting.orientation.values)
        g.axes[0][0].text(.2, .7, 
                                'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                horizontalalignment='left')

        # axis labels
        g.set_axis_labels('Color CS','Orientation CS', fontsize = 10, labelpad=15)
        g.axes[0][0].set(xlim=(.15, .75), ylim=(.15, .75))

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_correlations_CS-color_orientation.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )))

        # COLOR vs CONJUNCTION CS   
        g = sns.lmplot(data = corr_df4plotting, x = 'color', y = 'conjunction', height = 5)

        rho, pval = scipy.stats.spearmanr(corr_df4plotting.color.values, 
                                        corr_df4plotting.conjunction.values)
        g.axes[0][0].text(.2, .7, 
                                'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                horizontalalignment='left')

        # axis labels
        g.set_axis_labels('Color CS','Conjunction CS', fontsize = 10, labelpad=15)
        g.axes[0][0].set(xlim=(.15, .75), ylim=(.15, .75))

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_correlations_CS-color_conjunction.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )))

        # ORIENTATION vs CONJUNCTION CS   
        g = sns.lmplot(data = corr_df4plotting, x = 'conjunction', y = 'orientation', height = 5)

        rho, pval = scipy.stats.spearmanr(corr_df4plotting.conjunction.values, 
                                        corr_df4plotting.orientation.values)
        g.axes[0][0].text(.2, .7, 
                                'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                horizontalalignment='left')

        # axis labels
        g.set_axis_labels('Conjunction CS','Orientation CS', fontsize = 10, labelpad=15)
        g.axes[0][0].set(xlim=(.15, .75), ylim=(.15, .75))

        if save_fig:
            g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_correlations_CS-orientation_conjunction.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )))

        # Getting the Upper Triangle of the co-relation matrix
        #matrix = np.triu(corr_df4plotting.corr())
        w = sns.heatmap(corr_df4plotting.corr(method = 'spearman'), 
                    xticklabels=corr_df4plotting.corr().columns.values,
                    yticklabels=corr_df4plotting.corr().columns.values,
                    annot=True, cmap='Reds', vmin=0.6, vmax=1)#, mask=matrix)

        if save_fig:
            w.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_correlations_CS-heatmap.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session
                                                                                                    )))      


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