
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
    
    
    def __init__(self, BehObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        dataObj : object
            object from one of the classes defined in load_eye_data.X
            
        """
        
        # set results object to use later on
        self.BehObj = BehObj
        # if output dir not defined, then make it in derivates
        if outputdir is None:
            self.outputdir = op.join(self.BehObj.dataObj.derivatives_pth,'plots', 'eyetracking')
        else:
            self.outputdir = outputdir
            
        # number of participants to plot
        self.nr_pp = len(self.BehObj.dataObj.sj_num)

        # set font type for plots globally
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Helvetica'


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
        hRes = self.BehObj.dataObj.params['window_extra']['size'][0]
        vRes = self.BehObj.dataObj.params['window_extra']['size'][1]     

        ## participant trial info
        pp_trial_info = self.BehObj.dataObj.trial_info_df[self.BehObj.dataObj.trial_info_df['sj'] == 'sub-{pp}'.format(pp = participant)]

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
        target_ori = 'R' if target_ori_deg == self.BehObj.dataObj.params['stimuli']['ori_deg'] else 'L'

        distr_ori_deg = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_ori.values[0]
        # convert to LR labels
        distr_ori = ['R' if ori == self.BehObj.dataObj.params['stimuli']['ori_deg'] else 'L' for ori in distr_ori_deg]

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
        for i, pp in enumerate(self.BehObj.dataObj.sj_num):
            
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7), dpi=100, facecolor='w', edgecolor='k')

            #### Reaction Time distribution ####
            pt.RainCloud(data = df_trl_fixations[(df_trl_fixations['sj'] == 'sub-{sj}'.format(sj = pp))], 
                        x = 'set_size', y = 'nr_fixations', pointplot = True, hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax1)
            ax1.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax1.set_ylabel('# Fixations', fontsize = 15, labelpad=15)
            ax1.set_title('# Fixations Search sub-{sj}'.format(sj = pp), fontsize = 20)
            # set x ticks as integer
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax1.tick_params(axis='both', labelsize = 15)
            ax1.set_ylim(0,17)

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
            sns.pointplot(data = df_trl_fixations[(df_trl_fixations['sj'] == 'sub-{sj}'.format(sj = pp))],
                        x = 'set_size', y = 'mean_fix_dur', hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'], ax = ax2)
            ax2.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax2.set_ylabel('Fixation duration (s)', fontsize = 15, labelpad=15)
            ax2.set_title('Mean Fixation duration Search sub-{sj}'.format(sj = pp), fontsize = 20)
            # set x ticks as integer
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax2.tick_params(axis='both', labelsize = 15)
            ax2.set_ylim(0,.4)

            # quick fix for legen
            ax2.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                    title="Target ecc", fancybox=True)

            if save_fig:
                pp_folder = op.join(self.outputdir, 'sub-{sj}'.format(sj = pp))
                os.makedirs(pp_folder, exist_ok=True)

                fig.savefig(op.join(pp_folder, 'sub-{sj}_ses-{ses}_task-{task}_VSearch_fixations.png'.format(sj = pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))
                
        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(18,7), dpi=100, facecolor='w', edgecolor='k')

            #### Reaction Time distribution ####
            pt.RainCloud(data = df_mean_fixations, 
                        x = 'set_size', y = 'mean_fixations', pointplot = True, hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linecolor = 'grey',alpha = .5, dodge = True, saturation = 1, ax = ax1)
            ax1.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax1.set_ylabel('# Fixations', fontsize = 15, labelpad=15)
            ax1.set_title('Mean # Fixations Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax1.tick_params(axis='both', labelsize = 15)
            ax1.set_ylim(0,17)

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
            sns.pointplot(data = df_mean_fixations,
                        x = 'set_size', y = 'mean_fix_dur', hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'], ax = ax2)
            ax2.set_xlabel('Set Size', fontsize = 15, labelpad=15)
            ax2.set_ylabel('Fixation duration (s)', fontsize = 15, labelpad=15)
            ax2.set_title('Mean Fixation duration Search N = {nr}'.format(nr = self.nr_pp), fontsize = 20)
            # set x ticks as integer
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(self.BehObj.dataObj.set_size[x]))) 
            ax2.tick_params(axis='both', labelsize = 15)
            ax2.set_ylim(0,.4)

            # quick fix for legen
            ax2.legend(loc = 'lower right',fontsize=10, handles = [handleA, handleB, handleC], 
                    title="Target ecc", fancybox=True)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_VSearch_fixations.png'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session, 
                                                                                                            task = self.BehObj.dataObj.task_name)))

            ### plot main Fixation group figure ###
            
            # make fake dataframe, to fill with nans
            # so we force plot to be in continuous x-axis
            fake_ss = np.array([i for i in np.arange(int(self.BehObj.dataObj.set_size[-1])) if i not in self.BehObj.dataObj.set_size[:2]])

            tmp_df = pd.DataFrame({'sj': [], 'target_ecc': [], 'set_size': [], 'mean_RT': [], 'accuracy': []})
            for s in df_mean_fixations.sj.unique():
                
                tmp_df = pd.concat((tmp_df,
                                pd.DataFrame({'sj': np.repeat(s,len(np.repeat(fake_ss,3))), 
                                                'target_ecc': np.tile(self.BehObj.dataObj.ecc,len(fake_ss)), 
                                                'set_size': np.repeat(fake_ss,3), 
                                                'mean_fixations': np.repeat(np.nan,len(np.repeat(fake_ss,3))), 
                                                'mean_fix_dur': np.repeat(np.nan,len(np.repeat(fake_ss,3)))})))
            fake_DF = pd.concat((df_mean_fixations,tmp_df))

            ## actually plot
            fig, ax1 = plt.subplots(1,1, figsize=(9,3), dpi=1000, facecolor='w', edgecolor='k')

            pt.RainCloud(data = fake_DF, #df_mean_fixations, 
                        x = 'set_size', y = 'mean_fixations', pointplot = False, hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linewidth = 1, alpha = .9, dodge = True, saturation = 1, 
                        point_size = 1.8, width_viol = 3,
                        width_box = .72, offset = 0.45, move = .8, jitter = 0.25,
                        ax = ax1)
            ax1.set_xlabel('Set Size [items]', fontsize = 16, labelpad = 15)
            ax1.set_ylabel('# Fixations', fontsize = 16, labelpad = 15)
            #ax1.set_title('Reaction Times Crowding N = {nr}'.format(nr = self.nr_pp))
            ax1.set_ylim(0,17)
            ax1.set_xlim(5,33)

            ax1.tick_params(axis='both', labelsize=14)
            ax1.xaxis.set_ticks(np.arange(7, 33, 3))
            ax1.set_xticklabels([str(i) for i in np.arange(7, 33, 3)])

            #sns.pointplot(data = df_mean_results, 
            #             x = 'set_size', y = 'mean_RT', hue='target_ecc',
            #             palette = search_behaviour.dataObj.params['plotting']['ecc_colors'],
            #             linewidth = 1, alpha = .75, dodge = True, saturation = 1, 
            #             point_size = 1, ax = ax1)

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
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_Fixations.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')

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
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

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
                data = corr_df4plotting, x = 'critical_spacing', y = 'mean_fixations',markers = 'x',
                col = 'target_ecc', row = 'set_size', height = 4, hue = 'target_ecc',
                palette = [self.BehObj.dataObj.params['plotting']['crwd_type_colors'][crowding_type]],
                facet_kws = dict(sharex = True, sharey = True)
            )
            
            # main axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type.capitalize()), '# Fixations', fontsize = 18, labelpad=15)
            g.set(xlim=(.15, .71), ylim=(0, 18))

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
                ax.text(.8, 9, '{s} items'.format(s = self.BehObj.dataObj.set_size[ind]) , rotation = 0, fontsize = 22)

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
                for ss_ind, ss in enumerate(self.BehObj.dataObj.set_size):
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].mean_fixations.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                    g.axes[e_ind, ss_ind].text(.48, 15, 
                                            r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                                        '\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                                            horizontalalignment='left', fontsize = 16, weight='bold')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_Fix_correlation_CS_{ct}.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session,
                                                                                                ct = crowding_type)),
                        bbox_inches='tight')


            ## plot same but for mean fixation duration ###
            
            ## plot ecc x set size grid
            # of correlations

            g = sns.lmplot(
                data = corr_df4plotting, x = 'critical_spacing', y = 'mean_fix_dur',
                col = 'target_ecc', row = 'set_size', height = 3, #palette = 'flare',
                facet_kws = dict(sharex = True, sharey = True)
            )

            # main axis labels
            g.set_axis_labels('{ct} CS'.format(ct = crowding_type), 'Fixation duration (s)', fontsize = 10, labelpad=15)
            g.set(xlim=(.15, .75), ylim=(0, .4))

            ## set subplot titles
            for ax, title in zip(g.axes[0], ['Target {e} deg'.format(e = e_num) for e_num in self.BehObj.dataObj.ecc]):
                ax.set_title(title, fontsize = 15, pad = 25)

            # remove unecessary title
            for rax in g.axes[1:]:
                for cax in rax:
                    cax.set_title('')

            # add row title
            for ind, ax in enumerate([g.axes[i][-1] for i in range(len(self.BehObj.dataObj.set_size))]): # last column
                ax.text(.8, .2, '{s} items'.format(s = self.BehObj.dataObj.set_size[ind]) , rotation = 0, fontsize = 15)

            ## add Spearman correlation value and p-val 
            # as annotation
            for e_ind, ecc in enumerate(self.BehObj.dataObj.ecc):
                for ss_ind, ss in enumerate(self.BehObj.dataObj.set_size):
                    rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].mean_fix_dur.values, 
                                        corr_df4plotting[(corr_df4plotting['target_ecc'] == ecc) & \
                                    (corr_df4plotting['set_size'] == ss)].critical_spacing.values)

                    g.axes[e_ind, ss_ind].text(.2, .35, 
                                            'rho = %.2f \np-value = %.3f'%(rho,pval), 
                                            horizontalalignment='left')

            if save_fig:
                g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_correlations_DurFixations_CS-{ct}.png'.format(nr = self.nr_pp,
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)))

    def plot_fixDTC_search(self, df_mean_fix_on_DISTfeatures = None, save_fig = True):

        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            ### plot main ratio DTC group figure ###
            
            # make fake dataframe, to fill with nans
            # so we force plot to be in continuous x-axis
            fake_ss = np.array([i for i in np.arange(int(self.BehObj.dataObj.set_size[-1])) if i not in self.BehObj.dataObj.set_size[:2]])

            tmp_df = pd.DataFrame({'sj': [], 'target_ecc': [], 'set_size': [], 'mean_RT': [], 'accuracy': []})
            for s in df_mean_fix_on_DISTfeatures.sj.unique():
                
                tmp_df = pd.concat((tmp_df,
                                pd.DataFrame({'sj': np.repeat(s,len(np.repeat(fake_ss,3))), 
                                                'target_ecc': np.tile(self.BehObj.dataObj.num_ecc,len(fake_ss)), 
                                                'set_size': np.repeat(fake_ss,3), 
                                                'mean_RT': np.repeat(np.nan,len(np.repeat(fake_ss,3))), 
                                                'accuracy': np.repeat(np.nan,len(np.repeat(fake_ss,3)))})))
            fake_DF = pd.concat((df_mean_fix_on_DISTfeatures,tmp_df))

            fig, ax1 = plt.subplots(1,1, figsize=(9,3), dpi=1000, facecolor='w', edgecolor='k')

            # RATIO OF DISTRACTOR FIXATIONS ON TARGET COLOR
            pt.RainCloud(data = fake_DF, #df_mean_fix_on_DISTfeatures, 
                        x = 'set_size', y = 'mean_percent_fix_on_DTC', pointplot = False, hue='target_ecc',
                        palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                        linewidth = 1, alpha = .9, dodge = True, saturation = 1, 
                        point_size = 1.8, width_viol = 3,
                        width_box = .72, offset = 0.45, move = .8, jitter = 0.25,
                        ax = ax1)

            ax1.set_xlabel('Set Size [items]', fontsize = 16, labelpad = 15)
            ax1.set_ylabel('Ratio of Fixations on \nTarget-colored Distractors', fontsize = 16, labelpad = 10)
            # set x ticks as integer
            ax1.tick_params(axis='both', labelsize=14)
            #ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(search_behaviour.dataObj.set_size[x]))) 
            ax1.set_ylim(0.45,1.01)
            ax1.set_xlim(5,33)

            ax1.tick_params(axis='both', labelsize=14)
            ax1.xaxis.set_ticks(np.arange(7, 33, 3))
            ax1.set_xticklabels([str(i) for i in np.arange(7, 33, 3)])

            # add horizontal line indicating selectivity threshold
            ax1.axhline(.5, ls = '--', lw = 1, c = 'grey')

            # quick fix for legen
            handleA = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][4],
                                    label = '4 deg')
            handleB = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][8],
                                    label = '8 deg')
            handleC = mpatches.Patch(color = self.BehObj.dataObj.params['plotting']['ecc_colors'][12],
                                    label = '12 deg')
            ax1.legend(loc = 'lower right',fontsize=8, handles = [handleA, handleB, handleC], 
                                title="Target ecc")#, fancybox=True)

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_Fix_ratio_DTC.svg'.format(nr = self.nr_pp,
                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')

    def plot_correlations_fixDTC_slopes_search(self, df_mean_DTC_ecc = None, df_search_slopes = None, 
                                                    seed_num = 846, save_fig = True):

        """
        Plot correlation between color selectivity and search efficiency (linear slopes)
        per ecc
        """

        # if we have more than 1 participant data in object
        if self.nr_pp > 1: # make group plot

            # build tidy dataframe with relevant info
            corr_df4plotting = pd.DataFrame([])

            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):
                
                # loop over ecc
                for e in self.BehObj.dataObj.ecc:

                    # make temporary dataframe
                    tmp_df = df_mean_DTC_ecc[(df_mean_DTC_ecc['sj'] == 'sub-{s}'.format(s = pp)) &\
                                            (df_mean_DTC_ecc['target_ecc'] == e)]
                    tmp_df['search_slope'] = df_search_slopes[(df_search_slopes['sj']== 'sub-{s}'.format(s = pp)) &\
                                                            (df_search_slopes['target_ecc'] == e)].slope.values[0]
                    # append
                    corr_df4plotting = pd.concat((corr_df4plotting,
                                                tmp_df.copy()))

            ## actually plot

            # correlation scatter + linear regression
            g = sns.lmplot(data = corr_df4plotting, x = 'search_slope', y = 'mean_percent_fix_on_DTC',
                height = 4, markers = 'x', col= 'target_ecc',
                hue = 'target_ecc', palette = self.BehObj.dataObj.params['plotting']['ecc_colors'],
                facet_kws = dict(sharex = True, sharey = True)
            )

            # axis labels
            g.set_axis_labels('RT/set size (ms/item)', 'Ratio of Fixations on \nTarget-colored Distractors', 
                            fontsize = 18, labelpad=15)#, fontweight="bold")
            #g.set_ylabel('Ratio of Fixations on \nTarget-colored Distractors', fontsize = 16, labelpad = 10)
            g.set(xlim=(0, 130), ylim=(0.67, 1))

            ## set subplot titles
            for ax, title in zip(g.axes[0], ['{e} deg'.format(e = e) for e in self.BehObj.dataObj.ecc]):
                ax.set_title(title, fontsize = 22, pad = 30)
                ax.tick_params(axis='both', labelsize=14)

            # plot rho per ecc
            for ind, e in enumerate(self.BehObj.dataObj.ecc):
                rho, pval = scipy.stats.spearmanr(corr_df4plotting[corr_df4plotting['target_ecc'] == e].search_slope.values, 
                                                corr_df4plotting[corr_df4plotting['target_ecc'] == e].mean_percent_fix_on_DTC.values)

                g.axes[0,ind].text(120, .7, 
                            r"$\rho$ = {r}".format(r = '%.2f'%(rho)), #+\
                            #'\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                            horizontalalignment='right', fontsize = 16, weight='bold')

            if save_fig:
                g.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT_slope_correlation_Fix_ratio_DTC.svg'.format(nr = self.nr_pp,
                                                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')
                
            # permutate correlations and plot distribution
            fig, ax1 = plt.subplots(1,3, figsize=(13, 5), dpi=100, facecolor='w', edgecolor='k', sharey = True)

            for ind, e in enumerate(self.BehObj.dataObj.ecc):
                slope_arr = corr_df4plotting[corr_df4plotting['target_ecc'] == e].search_slope.values
                DTC_arr = corr_df4plotting[corr_df4plotting['target_ecc'] == e].mean_percent_fix_on_DTC.values

                # get observed correlation value
                rho, pval = scipy.stats.spearmanr(slope_arr, DTC_arr)

                # get permutation values
                perm_rho, pval_perm = utils.permutation_correlations(slope_arr, DTC_arr, method = 'spearman',
                                                                    perm_num=10000, seed = seed_num + ind,
                                                                    p_val_side='two-sided')

                ax1[ind].hist(perm_rho, color = self.BehObj.dataObj.params['plotting']['ecc_colors'][e], #'#7796a3', 
                                edgecolor='k', alpha=0.65,bins=50)
                ax1[ind].axvline(rho, color='black', linestyle='dashed', linewidth=1.5)
                ax1[ind].text(-.45, 550, 
                        #'observed\n'+\
                    # r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                        '\n\npermutation'+\
                    '\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval_perm))[1:],
                    horizontalalignment='left', fontsize = 14, weight='bold')

                ax1[ind].set_xlabel('Permutation '+r"$\rho$", fontsize=18, labelpad = 15)

            # axis labels
            ax1[0].set_ylabel('Frequency', fontsize=18, labelpad = 15)

            ## set subplot titles
            for ax, title in zip(ax1, ['{e} deg'.format(e = e) for e in self.BehObj.dataObj.ecc]):
                ax.set_title(title, fontsize = 22, pad = 30)
                ax.tick_params(axis='both', labelsize=14)
                ax.set(xlim=(-.5, .5))

            if save_fig:
                fig.savefig(op.join(self.outputdir, 'Nsj-{nr}_ses-{ses}_task-{task}_RT_slope_permutations_Fix_ratio_DTC.svg'.format(nr = self.nr_pp,
                                                                                                                                ses = self.BehObj.dataObj.session, 
                                                                                                                                task = self.BehObj.dataObj.task_name)),
                            bbox_inches='tight')

    def plot_participant_scanpath_search(self, participant, eye_events_df = None, save_fig = True):

        """
        Plot scanpath for ALL search trials of participant
        """

        # gabor radius in pixels
        r_gabor = (self.BehObj.dataObj.params['stimuli']['size_deg']/2)/self.BehObj.get_dva_per_pix(height_cm = self.BehObj.dataObj.params['monitor_extra']['height'], 
                                                                                                distance_cm = self.BehObj.dataObj.params['monitor']['distance'], 
                                                                                                vert_res_pix = self.BehObj.dataObj.params['window_extra']['size'][1])

        ## loop over blocks and trials
        for blk in eye_events_df.block_num.unique():
            for trl in eye_events_df[eye_events_df['block_num'] == blk].trial.unique():
                self.plot_search_saccade_path(participant, eye_events_df, 
                                                trial_num = trl, block_num = blk, 
                                                r_gabor = r_gabor, save_fig = save_fig)

    def plot_correlations_slopeNumFix_CS(self, df_CS = None, df_search_fix_slopes = None, 
                                        crowding_type_list = ['orientation', 'color', 'conjunction'],
                                        save_fig = True, outdir = None, seed_num = 24):
        
        """ plot correlations between num fixations SLOPEs and CS
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        # build tidy dataframe with relevant info
        corr_fix_slope_df4plotting = pd.DataFrame([])

        for crowding_type in crowding_type_list:
            # loop over subjects
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

                # make temporary dataframe
                tmp_df = df_search_fix_slopes[(df_search_fix_slopes['sj']== 'sub-{s}'.format(s = pp))]
                tmp_df['critical_spacing'] = df_CS[(df_CS['crowding_type']== crowding_type) & \
                                            (df_CS['sj']== 'sub-{s}'.format(s = pp))].critical_spacing.values[0]
                tmp_df['crowding_type'] = np.tile(crowding_type, len(tmp_df))

                # append
                corr_fix_slope_df4plotting = pd.concat((corr_fix_slope_df4plotting,
                                            tmp_df.copy()))
        
        ## plot correlations per crowding type 
        g = sns.lmplot(
            data = corr_fix_slope_df4plotting, x = 'critical_spacing', y = 'slope',
            col= 'crowding_type', height = 4, markers = 'x',
            hue = 'crowding_type', palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
            facet_kws = dict(sharex = True, sharey = True)
        )

        # axis labels
        g.set_axis_labels('CS', 'Fixations/set size (fix/item)', fontsize = 18, labelpad=15, fontweight="bold")
        g.set(xlim=(.15, .71), ylim=(0, .5))

        ## set subplot titles
        for ax, title in zip(g.axes[0], ['{ct}'.format(ct = c).capitalize() for c in df_CS.crowding_type.unique()]):
            ax.set_title(title, fontsize = 22, pad = 30)
            ax.tick_params(axis='both', labelsize=14)

        ## add Spearman correlation value and p-val 
        # as annotation
        p_val_all = []
        for ct_ind, ct in enumerate(df_CS.crowding_type.unique()):
            
            rho, pval = scipy.stats.spearmanr(corr_fix_slope_df4plotting[(corr_fix_slope_df4plotting['crowding_type'] == ct)].slope.values, 
                                        corr_fix_slope_df4plotting[(corr_fix_slope_df4plotting['crowding_type'] == ct)].critical_spacing.values)
            p_val_all.append(pval)

            g.axes[0][ct_ind].text(.65, .05, 
                        r"$\rho$ = {r}".format(r = '%.2f'%(rho)), #+\
                        #'\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                        horizontalalignment='right', fontsize = 16, weight='bold')

        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_Fix_slope_correlation_CS.svg'.format(nr = self.nr_pp,
                                                                                                    ses = self.BehObj.dataObj.session)),
                    bbox_inches='tight')
                
        ## permutate correlations and plot
        fig, ax1 = plt.subplots(1,3, figsize=(13, 5), dpi=100, facecolor='w', edgecolor='k', sharey=True)

        for ct_ind, ct in enumerate(df_CS.crowding_type.unique()):
            
            slope_arr = corr_fix_slope_df4plotting[(corr_fix_slope_df4plotting['crowding_type'] == ct)].slope.values
            cs_arr = corr_fix_slope_df4plotting[(corr_fix_slope_df4plotting['crowding_type'] == ct)].critical_spacing.values
            
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
        #ax1[0].set_xlabel('Permutation '+r"$\rho$", fontsize=18, labelpad = 15)
        ax1[0].set_ylabel('Frequency', fontsize=18, labelpad = 15)

        ## set subplot titles
        for ax, title in zip(ax1, ['{ct}'.format(ct = c).capitalize() for c in df_CS.crowding_type.unique()]):
            ax.set_title(title, fontsize = 22, pad = 30)
            ax.tick_params(axis='both', labelsize=14)
            ax.set(xlim=(-.5, .5))

        if save_fig:
            fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_Fix_slope_correlation_CS_permutation.svg'.format(nr = self.nr_pp,
                                                                                                            ses = self.BehObj.dataObj.session)),
                    bbox_inches='tight')
            
    def plot_correlations_RTFixRho_CS(self, df_CS = None, df_manual_responses = None, df_trl_fixations = None,
                                            save_fig = True, outdir = None, seed_num = 457):
        
        """ plot correlations between RT-fixations rho and CS
        
        Parameters
        ----------
        save_fig : bool
            save figure in output dir
            
        """

        ## merge RT df with trial fixations df
        joint_df = df_manual_responses[df_manual_responses['correct_response'] == 1].merge(df_trl_fixations, 
                                                                                        how='left')
        joint_df = joint_df.dropna()

        ## get the correlation between fixations and RT
        # across trials per participant
        # and combine with critical spacing values

        # build tidy dataframe with relevant info
        corr_df4plotting = pd.DataFrame([])

        for pp in self.BehObj.dataObj.sj_num:
            
            rho, pval = scipy.stats.spearmanr(joint_df[joint_df['sj'] == 'sub-{sj}'.format(sj = pp)].RT.values, 
                                        joint_df[joint_df['sj'] == 'sub-{sj}'.format(sj = pp)].nr_fixations.values)
            # append
            corr_df4plotting = pd.concat((corr_df4plotting,
                                df_CS[df_CS['sj'] == 'sub-{sj}'.format(sj = pp)].join(pd.DataFrame({'rho': np.tile(rho, 3)}))))

        ## actually plot
        g = sns.lmplot(data = corr_df4plotting, 
                        x = 'critical_spacing', y = 'rho', col = 'crowding_type', 
                        height = 4, markers = 'x',
                        hue = 'crowding_type', palette = self.BehObj.dataObj.params['plotting']['crwd_type_colors'],
                        facet_kws = dict(sharex = True, sharey = True))

        # axis labels
        g.set_axis_labels('CS', 'RT-Fixations '+r"$\rho$", fontsize = 18, labelpad=15)
        g.set(xlim=(.15, .71), ylim=(0.75, 1))

        # set subplot titles
        for ax, title in zip(g.axes[0], ['{ct}'.format(ct = c.capitalize()) for c in corr_df4plotting.crowding_type.unique()]):
            ax.set_title(title, fontsize = 22, pad = 30)
            
        for ind, c in enumerate(corr_df4plotting.crowding_type.unique()):
            rho, pval = scipy.stats.spearmanr(corr_df4plotting[(corr_df4plotting['crowding_type'] == c)].rho.values, 
                                    corr_df4plotting[(corr_df4plotting['crowding_type'] == c)].critical_spacing.values)

            g.axes[0][ind].text(.55, .8, #.48, .8, 
                                r"$\rho$ = {r}".format(r = '%.2f'%(rho)), #+\
                            #'\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval))[1:],
                                horizontalalignment='left', fontsize = 16, weight='bold')
            
            g.axes[0][ind].tick_params(axis='both', labelsize=14)
        
        if save_fig:
            g.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_RTFixRho_CS_correlations.svg'.format(nr = self.nr_pp,
                                                                                            ses = self.BehObj.dataObj.session)),
                    bbox_inches='tight')   
        
        ## permutate correlations and plot
        fig, ax1 = plt.subplots(1,3, figsize=(13, 5), dpi=100, facecolor='w', edgecolor='k', sharey = True)

        for ct_ind, ct in enumerate(df_CS.crowding_type.unique()):
            
            RTrho_arr = corr_df4plotting[(corr_df4plotting['crowding_type'] == ct)].rho.values
            cs_arr = corr_df4plotting[(corr_df4plotting['crowding_type'] == ct)].critical_spacing.values
            
            # get observed correlation value
            rho, pval = scipy.stats.spearmanr(RTrho_arr, cs_arr)
            
            # get permutation values
            perm_rho, pval_perm = utils.permutation_correlations(RTrho_arr, cs_arr, method = 'spearman',
                                                                perm_num=10000, seed = seed_num + ct_ind,
                                                                p_val_side='two-sided')
            
            ax1[ct_ind].hist(perm_rho, color = self.BehObj.dataObj.params['plotting']['crwd_type_colors'][ct], 
                            edgecolor='k', alpha=0.65,bins=50)
            ax1[ct_ind].axvline(rho, color='black', linestyle='dashed', linewidth=1.5)
            ax1[ct_ind].text(-.45, 550, 
                            #'observed\n'+\
                        #r"$\rho$ = {r}".format(r = '%.2f'%(rho))+\
                            '\n\npermutation'+\
                        '\n{p}'.format(p = '$\it{p}$ = ')+('%.3f'%(pval_perm))[1:],
                        horizontalalignment='left', fontsize = 14, weight='bold')
            ax1[ct_ind].set_xlabel('Permutation '+r"$\rho$", fontsize=18, labelpad = 15)
            
        # axis labels
        #ax1[0].set_xlabel('Permutation '+r"$\rho$", fontsize=18, labelpad = 15)
        ax1[0].set_ylabel('Frequency', fontsize=18, labelpad = 15)

        ## set subplot titles
        for ax, title in zip(ax1, ['{ct}'.format(ct = c).capitalize() for c in df_CS.crowding_type.unique()]):
            ax.set_title(title, fontsize = 22, pad = 30)
            ax.tick_params(axis='both', labelsize=14)
            ax.set(xlim=(-.5, .5))

        if save_fig:
            fig.savefig(op.join(outdir, 'Nsj-{nr}_ses-{ses}_RTFixRho_CS_permutation.svg'.format(nr = self.nr_pp,
                                                                                            ses = self.BehObj.dataObj.session)),
                    bbox_inches='tight')  
            


                                                                                                        
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
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

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
                                                                                                        ses = self.BehObj.dataObj.session,
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
                                                                                                        ses = self.BehObj.dataObj.session,
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
            for _, pp in enumerate(self.BehObj.dataObj.sj_num):

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
                                                                                                        ses = self.BehObj.dataObj.session,
                                                                                                        ct = crowding_type)))