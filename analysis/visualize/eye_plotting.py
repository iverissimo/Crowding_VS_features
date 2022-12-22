
from tty import CC
import numpy as np
import os
import os.path as op
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
import utils

import ptitprince as pt # raincloud plots
import matplotlib.patches as mpatches
from  matplotlib.ticker import FuncFormatter

class PlotsEye:
    
    
    def __init__(self, behObj, outputdir = None):
        
        """__init__
        constructor for class 
        
        Parameters
        ----------
        BehObj : BehResponses object
            object from one of the classes defined in behaviour.manual_responses
            
        """
        
        # set results object to use later on
        self.behObj = behObj
        # if output dir not defined, then make it in derivates
        if outputdir is None:
            self.outputdir = op.join(self.behObj.derivatives_pth,'plots')
        else:
            self.outputdir = outputdir
            
        # number of participants to plot
        self.nr_pp = len(self.behObj.sj_num)


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
        hRes = self.behObj.params['window_extra']['size'][0]
        vRes = self.behObj.params['window_extra']['size'][1]     

        ## participant trial info
        pp_trial_info = self.behObj.trial_info_df[self.behObj.trial_info_df['sj'] == 'sub-{pp}'.format(pp = participant)]

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
        target_ori = 'R' if target_ori_deg == self.behObj.params['stimuli']['ori_deg'] else 'L'

        distr_ori_deg = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_ori.values[0]
        # convert to LR labels
        distr_ori = ['R' if ori == self.behObj.params['stimuli']['ori_deg'] else 'L' for ori in distr_ori_deg]

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