
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

class PlotsEye:
    
    
    def __init__(self, dataObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        dataObj : object
            object from one of the classes defined in load_eye_data.X
            
        """
        
        # set results object to use later on
        self.dataObj = dataObj
        # if output dir not defined, then make it in derivates
        if outputdir is None:
            self.outputdir = op.join(self.dataObj.derivatives_pth,'plots', 'eyetracking')
        else:
            self.outputdir = outputdir
            
        # number of participants to plot
        self.nr_pp = len(self.dataObj.sj_num)


    def plot_search_saccade_path(self, participant, eye_events_df, 
                                trial_num = 0, block_num = 0, r_gabor = 50, save_fig = True):
        
        """ plot saccade path for visual search trials
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        outdir = op.join(self.outputdir, 'scanpaths', 'sub-{pp}'.format(pp = participant))
        os.makedirs(outdir, exist_ok=True)

        ## screen resolution
        hRes = self.dataObj.params['window_extra']['size'][0]
        vRes = self.dataObj.params['window_extra']['size'][1]     

        ## participant trial info
        pp_trial_info = self.dataObj.trial_info_df[self.dataObj.trial_info_df['sj'] == 'sub-{pp}'.format(pp = participant)]

        ## get target and distractor positions as strings in list
        target_pos = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].target_pos.values[0]
        distr_pos = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_pos.values[0]

        ## get target and distractor colors for plotting
        target_color = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].target_color.values[0]

        distr_color = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_color.values[0]

        ## get target and distractor orientations for plotting
        target_ori_deg = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].target_ori.values[0]
        # convert to LR labels
        target_ori = 'R' if target_ori_deg == self.dataObj.params['stimuli']['ori_deg'] else 'L'

        distr_ori_deg = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_ori.values[0]
        # convert to LR labels
        distr_ori = ['R' if ori == self.dataObj.params['stimuli']['ori_deg'] else 'L' for ori in distr_ori_deg]

        ## make figure
        f, s = plt.subplots(1, 1, figsize=(8,8))

        # add target
        s.add_artist(plt.Circle((target_pos[0], target_pos[1]), radius=r_gabor, 
                                facecolor=np.array(target_color)/255, fill=True, edgecolor='black', lw = 2))
        s.annotate(target_ori, xy=(target_pos[0], target_pos[1]), fontsize=10, ha='center', va='center',color='w', alpha=1)
        s.set_xlim([-hRes/2,hRes/2])
        s.set_ylim([-vRes/2,vRes/2])
        s.set_aspect('equal') # to not make circles elipses
        s.vlines(0, -vRes/2,vRes/2, alpha = .2)
        s.hlines(0, -hRes/2,hRes/2, alpha = .2)

        # add distractors
        for w in range(len(distr_pos)):
            s.add_artist(plt.Circle((distr_pos[w][0], distr_pos[w][1]), radius=r_gabor, 
                                    color=np.array(distr_color[w])/255, fill=True))
            s.annotate(distr_ori[w], xy=(distr_pos[w][0], distr_pos[w][1]), 
                    fontsize=10, ha='center', va='center',color='w', alpha=1)

        ## plot saccade direction
        # draw an arrow between every saccade start and ending

        trial_sacc_df = eye_events_df[(eye_events_df['block_num'] == block_num) & \
                                    (eye_events_df['phase_name'] == 'stim') & \
                                    (eye_events_df['trial'] == trial_num) & \
                                    (eye_events_df['eye_event'] == 'SAC')]

        # if there were saccades
        if not trial_sacc_df.empty:
            for _,row in trial_sacc_df.iterrows(): # iterate over dataframe
                
                # make positions compatible with display
                sx = row['x_pos'] - hRes/2 # start x pos
                ex = row['x_pos2'] - hRes/2 # end x pos
                sy = row['y_pos'] - vRes/2; sy = -sy #start y pos
                ey = row['y_pos2'] - vRes/2; ey = -ey #end y pos
            
                s.arrow(sx, sy, ex-sx, ey-sy, alpha=0.5, fc='k', 
                        fill=True, shape='full', width=10, head_width=20, head_starts_at_zero=False, overhang=0)

        f.savefig(op.join(outdir,'scanpath_block-{bn}_trial-{tn}.png'.format(bn = block_num, tn = trial_num)))


    def plot_fixations_search(self, df_trl_fixations = None,
                                    df_mean_fixations = None, save_fig = True):
        
        """ plot mean number of fixations and duration for search 
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """
        
        # loop over subjects
        for i, pp in enumerate(self.dataObj.sj_num):
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7), dpi=100, facecolor='w', edgecolor='k')

            #### Reaction Time distribution ####
            pt.RainCloud(data = df_trl_fixations[(df_trl_fixations['sj'] == 'sub-{sj}'.format(sj = pp))], 
                        x = 'set_size', y = 'nr_fixations', pointplot = True, hue='target_ecc',
                        palette = self.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax1)
            ax1.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax1.set_ylabel('# Fixations', fontsize = 15, labelpad=15)
            ax1.set_title('# Fixations Search sub-{sj}'.format(sj = pp), fontsize = 20)
            # set x ticks as integer
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.dataObj.set_size[x]))) 
            ax1.tick_params(axis='both', labelsize = 15)
            ax1.set_ylim(0,17)

            # quick fix for legen
            handleA = mpatches.Patch(color = self.dataObj.params['plotting']['ecc_colors'][4],
                                    label = 4)
            handleB = mpatches.Patch(color = self.dataObj.params['plotting']['ecc_colors'][8],
                                    label = 8)
            handleC = mpatches.Patch(color = self.dataObj.params['plotting']['ecc_colors'][12],
                                    label = 12)
            ax1.legend(loc = 'upper left',fontsize=12, handles = [handleA, handleB, handleC], 
                    title="Target ecc", fancybox=True)

            #### Accuracy ####
            sns.pointplot(data = df_trl_fixations[(df_trl_fixations['sj'] == 'sub-{sj}'.format(sj = pp))],
                        x = 'set_size', y = 'mean_fix_dur', hue='target_ecc',
                        palette = self.dataObj.params['plotting']['ecc_colors'], ax = ax2)
            ax2.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax2.set_ylabel('Fixation duration (s)', fontsize = 15, labelpad=15)
            ax2.set_title('Mean Fixation duration Search sub-{sj}'.format(sj = pp), fontsize = 20)
            # set x ticks as integer
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.dataObj.set_size[x]))) 
            ax2.tick_params(axis='both', labelsize = 15)
            ax2.set_ylim(0,.4)

            # quick fix for legen
            ax2.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                    title="Target ecc", fancybox=True)

            if save_fig:
                pp_folder = op.join(self.outputdir, 'sub-{sj}'.format(sj = pp))
                os.makedirs(pp_folder, exist_ok=True)

                fig.savefig(op.join(pp_folder, 'sub-{sj}_ses-{ses}_task-{task}_VSearch_fixations.png'.format(sj = pp,
                                                                                                            ses = self.dataObj.session, 
                                                                                                            task = self.dataObj.task_name)))
                
        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7), dpi=100, facecolor='w', edgecolor='k')

            #### Reaction Time distribution ####
            pt.RainCloud(data = df_mean_fixations, 
                        x = 'set_size', y = 'mean_fixations', pointplot = True, hue='target_ecc',
                        palette = self.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax1)
            ax1.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax1.set_ylabel('# Fixations', fontsize = 15, labelpad=15)
            ax1.set_title('Mean # Fixations Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.dataObj.set_size[x]))) 
            ax1.tick_params(axis='both', labelsize = 15)
            ax1.set_ylim(0,17)

            # quick fix for legen
            handleA = mpatches.Patch(color = self.dataObj.params['plotting']['ecc_colors'][4],
                                    label = 4)
            handleB = mpatches.Patch(color = self.dataObj.params['plotting']['ecc_colors'][8],
                                    label = 8)
            handleC = mpatches.Patch(color = self.dataObj.params['plotting']['ecc_colors'][12],
                                    label = 12)
            ax1.legend(loc = 'upper left',fontsize=12, handles = [handleA, handleB, handleC], 
                    title="Target ecc", fancybox=True)

            #### Accuracy ####
            sns.pointplot(data = df_mean_fixations,
                        x = 'set_size', y = 'mean_fix_dur', hue='target_ecc',
                        palette = self.dataObj.params['plotting']['ecc_colors'], ax = ax2)
            ax2.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax2.set_ylabel('Fixation duration (s)', fontsize = 15, labelpad=15)
            ax2.set_title('Mean Fixation duration Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.dataObj.set_size[x]))) 
            ax2.tick_params(axis='both', labelsize = 15)
            ax2.set_ylim(0,.4)

            # quick fix for legen
            ax2.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                    title="Target ecc", fancybox=True)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_VSearch_fixations.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.dataObj.session, 
                                                                                                            task = self.dataObj.task_name)))


    def plot_correlations_Fix_CS(self, df_CS = None, df_mean_fixations = None, 
                                        crowding_type_list = ['orientation', 'color', 'conjunction'],
                                        save_fig = True, outdir = None):
        
        """ plot correlations between mean number of fixations and CS
        
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
            for _, pp in enumerate(self.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_mean_fixations[(df_mean_fixations['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                    (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                
                # append
                corr_df4plotting = pd.concat((corr_df4plotting,
                                            tmp_df.copy()))

            ## plot ecc x set size grid
            # of correlations
            
            # facetgrid of plots
            g = sns.lmplot(
                data = corr_df4plotting, x = 'critical_spacing', y = 'mean_fixations',
                col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
                facet_kws = dict(sharex = True, sharey = True)
            )

            # main axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type), '# Fixations', fontsize = 10, labelpad=15)
            g.set(xlim=(.15, .75), ylim=(0, 18))

            ## set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.dataObj.ecc]):
                ax.set_title(title, fontsize = 15, pad = 25)

            # remove unecessary title
            for rax in g.axes[1:]:
                for cax in rax:
                    cax.set_title('')

            # add row title
            for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.dataObj.set_size))]): # last column
                ax.text(.8, 9, '{s} items'.format(s = self.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.dataObj.ecc):
                for ss_ind, ss in enumerate(self.dataObj.set_size):
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].mean_fixations.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                    g.axes[e_ind, ss_ind].text(.2, 15, 
                                            'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                            horizontalalignment='left')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_NumFixations_CS-{ct}.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.dataObj.session,
                                                                                                        ct = crowding_type)))

            ## plot same but for mean fixation duration
            
            ## plot ecc x set size grid
            # of correlations
            
            # facetgrid of plots
            g = sns.lmplot(
                data = corr_df4plotting, x = 'critical_spacing', y = 'mean_fix_dur',
                col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
                facet_kws = dict(sharex = True, sharey = True)
            )

            # main axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type), 'Fixation duration (s)', fontsize = 10, labelpad=15)
            g.set(xlim=(.15, .75), ylim=(0, .4))

            ## set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.dataObj.ecc]):
                ax.set_title(title, fontsize = 15, pad = 25)

            # remove unecessary title
            for rax in g.axes[1:]:
                for cax in rax:
                    cax.set_title('')

            # add row title
            for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.dataObj.set_size))]): # last column
                ax.text(.8, .2, '{s} items'.format(s = self.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.dataObj.ecc):
                for ss_ind, ss in enumerate(self.dataObj.set_size):
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].mean_fix_dur.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                    g.axes[e_ind, ss_ind].text(.2, .35, 
                                            'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                            horizontalalignment='left')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_DurFixations_CS-{ct}.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.dataObj.session,
                                                                                                        ct = crowding_type)))


        ############ also correlate with the mean CS (mean over crowding types) #############
        mean_df_CS = df_CS.groupby('sj').mean().reset_index()

        # build tidy dataframe with relevant info
        corr_df4plotting = pd.DataFrame([])

        # loop over subjects
        for _, pp in enumerate(self.dataObj.sj_num):

            # make temporary dataframe
            tmp_df = df_mean_fixations[(df_mean_fixations['sj']== 'sub-{s}'.format(s = pp))]
            tmp_df['critical_spacing'] = mean_df_CS[(mean_df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
            
            # append
            corr_df4plotting = pd.concat((corr_df4plotting,
                                        tmp_df.copy()))

        ## plot ecc x set size grid
        # of correlations
        
        # facetgrid of plots
        g = sns.lmplot(
            data = corr_df4plotting, x = 'critical_spacing', y = 'mean_fixations',
            col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
            facet_kws = dict(sharex = True, sharey = True)
        )

        # main axis labels
        g.set_axis_labels('Mean CS', '# Fixations', fontsize = 10, labelpad=15)
        g.set(xlim=(.15, .75), ylim=(0, 18))

        ## set subplot titles
        for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.dataObj.ecc]):
            ax.set_title(title, fontsize = 15, pad = 25)

        # remove unecessary title
        for rax in g.axes[1:]:
            for cax in rax:
                cax.set_title('')

        # add row title
        for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.dataObj.set_size))]): # last column
            ax.text(.8, 9, '{s} items'.format(s = self.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

        ## add Spearman correlation value and p-val 
        # as annotation
        for e_ind, ecc in enumerate(self.dataObj.ecc):
            for ss_ind, ss in enumerate(self.dataObj.set_size):
                rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                (corr_df4plotting['set_size'] == ss)].mean_fixations.values, 
                                    corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                g.axes[e_ind, ss_ind].text(.2, 15, 
                                        'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                        horizontalalignment='left')

        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_NumFixations_CS-mean.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.dataObj.session)))

        ## plot same but for mean fixation duration
        
        ## plot ecc x set size grid
        # of correlations
        
        # facetgrid of plots
        g = sns.lmplot(
            data = corr_df4plotting, x = 'critical_spacing', y = 'mean_fix_dur',
            col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
            facet_kws = dict(sharex = True, sharey = True)
        )

        # main axis labels
        g.set_axis_labels('Mean CS', 'Fixation duration (s)', fontsize = 10, labelpad=15)
        g.set(xlim=(.15, .75), ylim=(0, .4))

        ## set subplot titles
        for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.dataObj.ecc]):
            ax.set_title(title, fontsize = 15, pad = 25)

        # remove unecessary title
        for rax in g.axes[1:]:
            for cax in rax:
                cax.set_title('')

        # add row title
        for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.dataObj.set_size))]): # last column
            ax.text(.8, .2, '{s} items'.format(s = self.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

        ## add Spearman correlation value and p-val 
        # as annotation
        for e_ind, ecc in enumerate(self.dataObj.ecc):
            for ss_ind, ss in enumerate(self.dataObj.set_size):
                rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                (corr_df4plotting['set_size'] == ss)].mean_fix_dur.values, 
                                    corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                g.axes[e_ind, ss_ind].text(.2, .35, 
                                        'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                        horizontalalignment='left')

        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_DurFixations_CS-mean.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.dataObj.session)))



    def plot_correlations_slopeNumFix_CS(self, df_CS = None, df_search_fix_slopes = None, 
                                        crowding_type_list = ['orientation', 'color', 'conjunction'],
                                        save_fig = True, outdir = None):
        
        """ plot correlations between num fixations SLOPEs and CS
        
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
            for _, pp in enumerate(self.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_search_fix_slopes[(df_search_fix_slopes['sj']== 'sub-{s}'.format(s = pp))]
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
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type), 'Fixations/set size (fix/item)', fontsize = 10, labelpad=15)
            g.set(xlim=(.15, .75), ylim=(0, .5))

            # set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.dataObj.ecc]):
                ax.set_title(title, fontsize = 15, pad = 25)
                

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.dataObj.ecc):
                rho, pval = scipy.stats.spearmanr(corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].slope.values, 
                                        corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].critical_spacing.values)

                g.axes[0][e_ind].text(.2, .4, 
                                'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                horizontalalignment='left')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_NumFixationsSlope_CS-{ct}.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.dataObj.session,
                                                                                                        ct = crowding_type)))

        ############ also correlate with the mean CS (mean over crowding types) #############
        mean_df_CS = df_CS.groupby('sj').mean().reset_index()

        # build tidy dataframe with relevant info
        corr_slope_df4plotting = pd.DataFrame([])

        # loop over subjects
        for _, pp in enumerate(self.dataObj.sj_num):

            # make temporary dataframe
            tmp_df = df_search_fix_slopes[(df_search_fix_slopes['sj']== 'sub-{s}'.format(s = pp))]
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
        g.set_axis_labels('Mean CS', 'Fixations/set size (fix/item)', fontsize = 10, labelpad=15)
        g.set(xlim=(.15, .75), ylim=(0, .5))

        # set subplot titles
        for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.dataObj.ecc]):
            ax.set_title(title, fontsize = 15, pad = 25)
            

        ## add Spearman correlation value and p-val 
        # as annotation
        for e_ind, ecc in enumerate(self.dataObj.ecc):
            rho, pval = scipy.stats.spearmanr(corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].slope.values, 
                                    corr_slope_df4plotting[(corr_slope_df4plotting['target_ecc'] == ecc)].critical_spacing.values)

            g.axes[0][e_ind].text(.2, .4, 
                            'rho = %.2f \np-value = %.3f'%(rho,pval), 
                            horizontalalignment='left')

        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_NumFixationsSlope_CS-mean.png'.format(nr = self.nr_pp,
                                                                                                    ses = self.dataObj.session)))

                                                                                                        
    def plot_correlations_Fix_CS_heatmap(self, df_CS = None, df_mean_fixations = None, 
                                        method = 'pearson', BehObj = None,
                                        crowding_type_list = ['orientation', 'color', 'conjunction'],
                                        save_fig = True, outdir = None):
        
        """ plot correlations between mean number of fixations and CS
        
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
            for _, pp in enumerate(self.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_mean_fixations[(df_mean_fixations['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                    (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                
                # append
                corr_df4plotting = pd.concat((corr_df4plotting,
                                            tmp_df.copy()))
    
            # get correlation dfs for each case
            # but need to replace column names
            
            corr_df, pval_df = BehObj.make_search_CS_corr_2Dmatrix(corr_df4plotting.rename(columns={'mean_fixations': 'y_val', 
                                                                                                'critical_spacing': 'x_val'}), 
                                                                    method = method)

            ## make figure
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,3), dpi=100, facecolor='w', edgecolor='k')

            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-.35, vmax=.35, ax=ax1)
            ax1.set_title('{mt} correlation'.format(mt = method), fontsize = 10)

            sns.heatmap(pval_df, annot=True, vmin=0.00, vmax=0.10, cmap='RdGy', ax=ax2)
            ax2.set_title('{mt} p-values'.format(mt = method), fontsize = 10)

            if save_fig:
                fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_{mt}_correlations_NumFixations_CS-{ct}_heatmap.png'.format(nr = self.nr_pp,
                                                                                                        mt = method,
                                                                                                        ses = self.dataObj.session,
                                                                                                        ct = crowding_type)))

            ## plot same but for mean fixation duration
            
            # get correlation dfs for each case
            # but need to replace column names
            
            corr_df, pval_df = BehObj.make_search_CS_corr_2Dmatrix(corr_df4plotting.rename(columns={'mean_fix_dur': 'y_val', 
                                                                                                'critical_spacing': 'x_val'}), 
                                                                    method = method)

            ## make figure
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9,3), dpi=100, facecolor='w', edgecolor='k')

            sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-.35, vmax=.35, ax=ax1)
            ax1.set_title('{mt} correlation'.format(mt = method), fontsize = 10)

            sns.heatmap(pval_df, annot=True, vmin=0.00, vmax=0.10, cmap='RdGy', ax=ax2)
            ax2.set_title('{mt} p-values'.format(mt = method), fontsize = 10)

            if save_fig:
                fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_{mt}_correlations_DurFixations_CS-{ct}_heatmap.png'.format(nr = self.nr_pp,
                                                                                                        mt = method,
                                                                                                        ses = self.dataObj.session,
                                                                                                        ct = crowding_type)))


    def plot_correlations_slopeNumFix_CS_heatmap(self, df_CS = None, df_search_fix_slopes = None, 
                                                    method = 'pearson', BehObj = None,
                                                    crowding_type_list = ['orientation', 'color', 'conjunction'],
                                                    save_fig = True, outdir = None):
                    
        """ plot correlations between num fixations SLOPEs and CS
        
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
            for _, pp in enumerate(self.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_search_fix_slopes[(df_search_fix_slopes['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                            (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                
                # append
                corr_slope_df4plotting = pd.concat((corr_slope_df4plotting,
                                            tmp_df.copy()))


            # get correlation dfs for each case
            # but need to replace column names
            
            corr_df, pval_df = BehObj.make_search_CS_corr_1Dmatrix(corr_slope_df4plotting.rename(columns={'slope': 'y_val', 
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
                fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_{mt}_correlations_NumFixationsSlope_CS-{ct}_heatmap.png'.format(nr = self.nr_pp,
                                                                                                        mt = method,
                                                                                                        ses = self.dataObj.session,
                                                                                                        ct = crowding_type)))