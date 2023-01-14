import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml

from analysis.behaviour import load_beh_data
from scipy.signal import savgol_filter
from scipy import spatial

import utils

import re
from sklearn.linear_model import LinearRegression

class EyeTrack:

    def __init__(self, dataObj):  # initialize child class

        """ Initializes EyeTrack object. 

        Parameters
        ----------
        dataObj: object
            task behavior object from  load_beh_data.X 
        """

        self.dataObj = dataObj
        self.outdir = op.join(self.dataObj.derivatives_pth, 'eyetracking')

        # gabor radius in pixels
        self.r_gabor = (self.dataObj.params['stimuli']['size_deg']/2)/self.get_dva_per_pix(height_cm = self.dataObj.params['monitor_extra']['height'], 
                                                                                            distance_cm = self.dataObj.params['monitor']['distance'], 
                                                                                            vert_res_pix = self.dataObj.params['window_extra']['size'][1])

        self.hRes = self.dataObj.params['window_extra']['size'][0]
        self.vRes = self.dataObj.params['window_extra']['size'][1]  

    def convert2asc(self, replace_val = 0.0001):
        
        """
        Convert edf files to asc samples and message files
        """
        
        EYEsamples_files = {}
        EYEevents_files = {}

        for i, pp in enumerate(self.dataObj.sj_num):

            # subject data folder
            sj_pth = op.join(op.join(self.dataObj.sourcedata_pth, 'sub-{sj}'.format(sj = pp)))

            # load trial info, with step up for that session
            edf_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.dataObj.session)) and \
                            'task-{task}'.format(task = self.dataObj.task_name) in x and x.endswith('.edf')]
            
            # samples file name
            sample_filename = edf_files[0].replace('.edf','_EYE_samples.asc')
            # events filename
            events_filename = edf_files[0].replace('.edf','_EYE_events.asc')
            
            ## convert to asc
            if not op.isfile(sample_filename):

                ## samples
                os.system("edf2asc -t -ftime -y -z -v -s -miss {val} -vel {file}".format(file = edf_files[0],
                                                                                        val = replace_val))
                os.system('mv {file} {new_file}'.format(file = edf_files[0].replace('.edf','.asc'),
                                                    new_file = sample_filename))
            # save absolute file names in dict
            EYEsamples_files['sub-{sj}'.format(sj = pp)] = sample_filename

            if not op.isfile(events_filename):
                ## events
                os.system("edf2asc -t -ftime -y -z -v -e {file}".format(file = edf_files[0]))
                os.system('mv {file} {new_file}'.format(file = edf_files[0].replace('.edf','.asc'),
                                                    new_file = events_filename))
            # save absolute file names in dict
            EYEevents_files['sub-{sj}'.format(sj = pp)] = events_filename


        self.EYEsamples_files = EYEsamples_files
        self.EYEevents_files = EYEevents_files

    
    def get_eyelink_events(self, filename, sampling_rate = 1000, save_as = None):
        
        """
        Get fixations and saccades for task
        as defined by eyelink criteria
        """

        ## read all lines from events file
        with open(filename) as file:
            lines = [line.rstrip() for line in file]

        ## data frame to store values
        eye_events_df = pd.DataFrame({'block_num': [], 'trial': [], 'phase': [], 'phase_name':[], 'phase_start_sample': [],
                                'eye_event': [], 'ev_start_sample': [], 'ev_end_sample': [], 'dur_in_sec': [], 
                                'x_pos': [], 'y_pos': [], 'x_pos2': [], 'y_pos2': [], 'eye': []})

        # block number counter
        block_num = -1

        # get start trial identifier
        # string and ind
        # NOTE each trial has different phases depending on stim, iti etc
        trl_msg = [[int(ind), str(val)] for ind, val in enumerate(lines) if 'start_type-stim_trial' in val and 'MSG' in val]

        # go through ind
        for ind in np.arange(np.array(trl_msg).shape[0]):
            
            # get trial phase identifier
            trial_id_str = lines[trl_msg[ind][0]].split('\t')[-1]

            # get trial number, phase number etc
            trial_num = int(re.findall('trial-\d{0,4}', trial_id_str)[0].split('-')[-1])
            phase_num = int(re.findall('phase-\d{0,4}', trial_id_str)[0].split('-')[-1])

            # phase name depends if start of block or not
            if trial_num == 0:
                if phase_num == 0:
                    phase_name = 'block_start'
                    block_num += 1
                elif (phase_num == 1) or (phase_num == 3):
                    phase_name = 'iti'
                elif phase_num == 2:
                    phase_name = 'stim'
            else:
                if phase_num == 0:
                    phase_name = 'stim'
                else:
                    phase_name = 'iti'

            # sample to time stamp beggining of trial phase
            phase_start_sample = float(lines[trl_msg[ind][0]].split('\t')[1])

            # get all eye events for this phase of trial
            if ind == np.array(trl_msg).shape[0]-1:
                eye_trial_phase = lines[trl_msg[ind][0]+1:]
            else:
                eye_trial_phase = lines[trl_msg[ind][0]+1:trl_msg[ind+1][0]-1]
                
            ## in case its last phase of a block,
            # subselect because there might be other events there
            end_block_ind = [w for w, w_str in enumerate(eye_trial_phase) if 'MSG' in w_str]
            # end of block indice
            if len(end_block_ind) > 0: 
                eye_trial_phase = eye_trial_phase[:end_block_ind[0]] 

            ## now fill data frame with all eye events for trial phase
            for ev in eye_trial_phase:

                if (ev[0] == 'E') and ('END' not in ev): # only looking at end events, because those have relevant info

                    ev_list = ev.split('\t') # split into list

                    if (float(ev_list[2]) >= phase_start_sample) and \
                        (('FIX' in ev_list[0]) or ('SAC' in ev_list[0])): # if event happens after trial phase started
                        
                        ## fixations
                        if 'FIX' in ev_list[0]:
                            try:
                                eye_event_name = 'FIX'
                                eye_used = ev_list[1]
                                ev_start_sample = float(ev_list[2]) 
                                ev_end_sample = float(ev_list[3]) 
                                dur_in_sec = float(ev_list[4])/sampling_rate
                                x_pos =  float(ev_list[5]) 
                                y_pos =  float(ev_list[6]) 
                                x_pos2 = np.nan
                                y_pos2 = np.nan

                            except ValueError:
                                print('Skipping Fixation of trial {it}, block {ib} because of missing values'.format(it = trial_num,
                                                                                                                    ib = block_num))
                                print(ev_list)

                        ## saccades
                        elif 'SAC' in ev_list[0]:
                            try:
                                eye_event_name = 'SAC'
                                eye_used = ev_list[1]
                                ev_start_sample = float(ev_list[2]) 
                                ev_end_sample = float(ev_list[3]) 
                                dur_in_sec = float(ev_list[4])/sampling_rate
                                x_pos =  float(ev_list[5]) 
                                y_pos =  float(ev_list[6]) 
                                x_pos2 = float(ev_list[7]) 
                                y_pos2 = float(ev_list[8]) 
                            
                            except ValueError:
                                print('Skipping Saccade of trial {it}, block {ib} because of missing values'.format(it = trial_num,
                                                                                                                    ib = block_num))
                                print(ev_list)

                        ## concat 
                        eye_events_df = pd.concat((eye_events_df, pd.DataFrame({'trial': [trial_num], 'block_num': [block_num],
                                                            'phase': [phase_num], 'phase_name':[phase_name], 
                                                            'phase_start_sample': [phase_start_sample],
                                                            'eye_event': [eye_event_name], 
                                                            'ev_start_sample': [ev_start_sample], 'ev_end_sample': [ev_end_sample], 
                                                            'dur_in_sec': [dur_in_sec], 'eye': [eye_used],
                                                            'x_pos': [x_pos], 'y_pos': [y_pos], 'x_pos2': [x_pos2], 'y_pos2': [y_pos2]})
                                        ), ignore_index=True)

        # save dataframe if absolute filename provided
        if save_as is not None:
            eye_events_df.to_csv(save_as, index=False)

        return eye_events_df

    def get_dva_per_pix(self, height_cm = 30, distance_cm = 73, vert_res_pix = 1080):

        """ calculate degrees of visual angle per pixel, 
        to use for screen boundaries when plotting/masking
        Parameters
        ----------
        height_cm : int
            screen height
        distance_cm: float
            screen distance (same unit as height)
        vert_res_pix : int
            vertical resolution of screen
        
        Outputs
        -------
        deg_per_px : float
            degree (dva) per pixel
        
        """

        # screen size in degrees / vertical resolution
        deg_per_px = (2.0 * np.degrees(np.arctan(height_cm /(2.0*distance_cm))))/vert_res_pix

        return deg_per_px 


class EyeTrackVsearch(EyeTrack):

    def __init__(self, dataObj):  # initialize child class

        """ Initializes EyeTrack object. 

        Parameters
        ----------
        dataObj: object
            task behavior object from  load_beh_data.X 
        """

        # need to initialize parent class (EyeTrack), indicating output infos
        super().__init__(dataObj = dataObj)

        # convert edf files, in case they have not been converted yet
        self.convert2asc()

    
    def get_search_mean_fixations(self, df_manual_responses = None, exclude_target_fix = True):

        """ 
        Get mean number of fixations and mean fixation durations
        for each participant, set size and ecc
        of search task
        """

        # make out dir, to save eye events dataframe
        os.makedirs(self.outdir, exist_ok=True)

        # set up empty df
        df_mean_fixations = pd.DataFrame({'sj': [],'target_ecc': [], 'set_size': [], 'mean_fixations': [], 'mean_fix_dur': []})

        ## loop over participants

        for pp in self.dataObj.sj_num:

            ## get all eye events (fixation and saccades)

            eye_df_filename = op.join(self.outdir, 'sub-{sj}_eye_events.csv'.format(sj = pp))
            
            if not op.isfile(eye_df_filename):
                print('Getting eye events for sub-{sj}'.format(sj = pp))
                eye_events_df = self.get_eyelink_events(self.EYEevents_files['sub-{sj}'.format(sj = pp)], 
                                                        sampling_rate = 1000, 
                                                        save_as = eye_df_filename)
            else:
                print('Loading %s'%eye_df_filename)
                eye_events_df = pd.read_csv(eye_df_filename)

            ## participant trial info
            pp_trial_info = self.dataObj.trial_info_df[self.dataObj.trial_info_df['sj'] == 'sub-{pp}'.format(pp = pp)]

            ## select only correct trials for participant
            pp_manual_response_df = df_manual_responses[(df_manual_responses['correct_response'] == 1) & \
                                                        (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]

            ## select only fixations and times when 
            # stimuli was displayed
            all_fixation_df = eye_events_df[(eye_events_df['eye_event'] == 'FIX') & \
                                        (eye_events_df['phase_name'] == 'stim')]

            # get number of blocks
            nr_blocks = all_fixation_df.block_num.unique().astype(int)

            # loop over ecc
            for e in self.dataObj.ecc:
                
                # loop over set size
                for ss in self.dataObj.set_size:
                    
                    fix = []
                    dur = []
                
                    for blk in nr_blocks:
                        # get trial indices for block
                        blk_trials = pp_manual_response_df[(pp_manual_response_df['block_num'] == blk) & \
                                                        (pp_manual_response_df['target_ecc'] == e) & \
                                                        (pp_manual_response_df['set_size'] == ss)].trial_num.values
                
                        for t in blk_trials:
                
                            ## fixations for this trial
                            trl_fix_df = all_fixation_df[(all_fixation_df['block_num'] == blk) & \
                                                        (all_fixation_df['trial'] == t)]

                            # if no fixations on trial to begin with
                            if trl_fix_df.empty:
                                dur.append(np.nan)
                                fix.append(0)
                            else:
                                if exclude_target_fix: # if we want to exclude fixations on target
                                    ## get target coordinates
                                    target_pos = pp_trial_info[(pp_trial_info['block'] == blk) & \
                                                            (pp_trial_info['index'] == t)].target_pos.values[0]

                                    # get x,y coordinates of last fixation
                                    fix_x = trl_fix_df.iloc[-1]['x_pos'] - self.hRes/2 # start x pos
                                    fix_y = trl_fix_df.iloc[-1]['y_pos'] - self.vRes/2; fix_y = -fix_y #start y pos

                                    # if fixation on target, remove
                                    if ((target_pos[0] - fix_x)**2 + (target_pos[-1] - fix_y)**2) <= self.r_gabor ** 2:
                                        trl_fix_df = trl_fix_df.iloc[:-1] 

                                # if no fixations in trial
                                if trl_fix_df.empty:
                                    dur.append(np.nan)
                                    fix.append(0)
                                else:
                                    # get fixation duration
                                    trl_fix_dur = trl_fix_df.dur_in_sec.values

                                    dur.append(np.nanmean(trl_fix_dur)) # append mean duration
                                    fix.append(len(trl_fix_dur)) # append number of fixations in each trial

                    # append in dataframe
                    df_mean_fixations = pd.concat((df_mean_fixations,
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],
                                                                'target_ecc': [e], 
                                                                'set_size': [ss], 
                                                                'mean_fixations': [np.nanmean(fix)], 
                                                                'mean_fix_dur': [np.nanmean(dur)]})), ignore_index = True)

        self.df_mean_fixations = df_mean_fixations


    def get_search_trl_fixations(self, df_manual_responses = None, exclude_target_fix = True):

        """ 
        Get number of fixations and mean fixation durations
        for each trial of search task
        """

        # make out dir, to save eye events dataframe
        os.makedirs(self.outdir, exist_ok=True)

        # set up empty df
        df_trl_fixations = pd.DataFrame({'sj': [], 'trial_num': [], 'block_num': [],
                                 'target_ecc': [], 'set_size': [], 'nr_fixations': [], 'mean_fix_dur': []})

        ## loop over participants

        for pp in self.dataObj.sj_num:

            ## get all eye events (fixation and saccades)

            eye_df_filename = op.join(self.outdir, 'sub-{sj}_eye_events.csv'.format(sj = pp))
            
            if not op.isfile(eye_df_filename):
                print('Getting eye events for sub-{sj}'.format(sj = pp))
                eye_events_df = self.get_eyelink_events(self.EYEevents_files['sub-{sj}'.format(sj = pp)], 
                                                        sampling_rate = 1000, 
                                                        save_as = eye_df_filename)
            else:
                print('Loading %s'%eye_df_filename)
                eye_events_df = pd.read_csv(eye_df_filename)

            ## participant trial info
            pp_trial_info = self.dataObj.trial_info_df[self.dataObj.trial_info_df['sj'] == 'sub-{pp}'.format(pp = pp)]

            ## select only correct trials for participant
            pp_manual_response_df = df_manual_responses[(df_manual_responses['correct_response'] == 1) & \
                                                        (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]

            ## select only fixations and times when 
            # stimuli was displayed
            all_fixation_df = eye_events_df[(eye_events_df['eye_event'] == 'FIX') & \
                                        (eye_events_df['phase_name'] == 'stim')]

            # get number of blocks
            nr_blocks = all_fixation_df.block_num.unique().astype(int)

            # loop over ecc
            for e in self.dataObj.ecc:
                
                # loop over set size
                for ss in self.dataObj.set_size:

                    for blk in nr_blocks:
                        # get trial indices for block
                        blk_trials = pp_manual_response_df[(pp_manual_response_df['block_num'] == blk) & \
                                                        (pp_manual_response_df['target_ecc'] == e) & \
                                                        (pp_manual_response_df['set_size'] == ss)].trial_num.values
                
                        for t in blk_trials:

                            ## fixations for this trial
                            trl_fix_df = all_fixation_df[(all_fixation_df['block_num'] == blk) & \
                                                        (all_fixation_df['trial'] == t)]

                            # if no fixations on trial to begin with
                            if trl_fix_df.empty:
                                dur = np.nan
                                fix = 0
                            else:
                                if exclude_target_fix: # if we want to exclude fixations on target
                                    ## get target coordinates
                                    target_pos = pp_trial_info[(pp_trial_info['block'] == blk) & \
                                                            (pp_trial_info['index'] == t)].target_pos.values[0]

                                    # get x,y coordinates of last fixation
                                    fix_x = trl_fix_df.iloc[-1]['x_pos'] - self.hRes/2 # start x pos
                                    fix_y = trl_fix_df.iloc[-1]['y_pos'] - self.vRes/2; fix_y = -fix_y #start y pos
                                    
                                    # if fixation on target, remove
                                    if ((target_pos[0] - fix_x)**2 + (target_pos[-1] - fix_y)**2) <= self.r_gabor ** 2:
                                        trl_fix_df = trl_fix_df.iloc[:-1] 

                                # if no fixations in trial
                                if trl_fix_df.empty:
                                    dur = np.nan
                                    fix = 0
                                else:
                                    # get fixation duration
                                    trl_fix_dur = trl_fix_df.dur_in_sec.values

                                    dur = np.nanmean(trl_fix_dur) # mean duration
                                    fix = len(trl_fix_dur) #  number of fixations in each trial

                            # append in dataframe
                            df_trl_fixations = pd.concat((df_trl_fixations,
                                                        pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],
                                                                    'trial_num': [t], 
                                                                    'block_num': [blk],
                                                                    'target_ecc': [e], 
                                                                    'set_size': [ss], 
                                                                    'nr_fixations': [fix], 
                                                                    'mean_fix_dur': [dur]})), ignore_index = True)

        self.df_trl_fixations = df_trl_fixations


    def get_fix_slopes(self, df_trl_fixations, fix_nr = True):

        """
        calculate search slopes and intercept per ecc

        Parameters
        ----------
        df_manual_responses : DataFrame
            dataframe with results from get_RTs()
        
        """ 

        # set empty df
        df_search_fix_slopes = pd.DataFrame({'sj': [],'target_ecc': [],  'slope': [], 'intercept': []})

        # loop over subjects
        for pp in self.dataObj.sj_num:

            print('calculating search slopes for sub-{sj}'.format(sj = pp))

            # loop over ecc
            for e in self.dataObj.ecc: 

                # sub-select df
                df_temp = df_trl_fixations[(df_trl_fixations['sj'] == 'sub-{sj}'.format(sj = pp)) & \
                                    (df_trl_fixations['target_ecc'] == e)]

                # fit linear regressor
                regressor = LinearRegression()
                if fix_nr:
                    regressor.fit(df_temp[['set_size']], df_temp[['nr_fixations']]) # slope in #fix/item
                else:
                    regressor.fit(df_temp[['set_size']], df_temp[['mean_fix_dur']]*1000) # because we want slope to be in ms/item

                # save df
                df_search_fix_slopes = pd.concat([df_search_fix_slopes, 
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)], 
                                                            'target_ecc': [e],   
                                                            'slope': [regressor.coef_[0][0]],
                                                            'intercept': [regressor.intercept_[0]]})])

        return df_search_fix_slopes 

    def get_trl_percent_fix_on_features(self, eye_events_df, participant = None, block_num = None, trial_num = None,
                                            hRes = 1920, vRes = 1080):

        """ 
        For a specific participant and trial,
        get the percentage of fixations that fall on distractors
        with the same color or orientation of target
        also outputs percentage of fixations on target itself
        """

        ## participant trial info
        pp_trial_info = self.dataObj.trial_info_df[self.dataObj.trial_info_df['sj'] == 'sub-{pp}'.format(pp = participant)]

        ## get target and distractor positions as strings in list
        target_pos = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].target_pos.values[0]
        distr_pos = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_pos.values[0]

        ## get target and distractor colors 
        target_color = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].target_color.values[0]

        distr_color = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_color.values[0]

        ## get target and distractor orientations 
        target_ori_deg = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].target_ori.values[0]
        # convert to LR labels
        target_ori = 'R' if target_ori_deg == self.dataObj.params['stimuli']['ori_deg'] else 'L'

        distr_ori_deg = pp_trial_info[(pp_trial_info['block'] == block_num) & \
                                (pp_trial_info['index'] == trial_num)].distractor_ori.values[0]
        # convert to LR labels
        distr_ori = ['R' if ori == self.dataObj.params['stimuli']['ori_deg'] else 'L' for ori in distr_ori_deg]

        ## get trial fixations
        trial_fix_df = eye_events_df[(eye_events_df['block_num'] == block_num) & \
                                    (eye_events_df['phase_name'] == 'stim') & \
                                    (eye_events_df['trial'] == trial_num) & \
                                    (eye_events_df['eye_event'] == 'FIX')]

        ## append distractor + target position 
        all_stim_pos = distr_pos + [target_pos]

        # kd-tree for quick nearest-neighbor lookup
        tree = spatial.KDTree(all_stim_pos)

        fix_target_color = 0
        fix_target_ori = 0
        fix_target = 0

        # if there were fixations
        if not trial_fix_df.empty:
            for _,row in trial_fix_df.iterrows(): # iterate over dataframe

                # make positions compatible with display
                fx = row['x_pos'] - hRes/2 # start x pos
                fy = row['y_pos'] - vRes/2; fy = -fy #start y pos
                #print('x: %s, y: %s'%(str(fx), str(fy)))
                
                # get nearest stim distance and index
                dist, stim_ind = tree.query([(fx,fy)])
                
                if stim_ind[0] < len(distr_pos): # if closest item is not target
                    #print('distance to closest item is %.2f'%dist)
                    
                    if distr_ori[stim_ind[0]] == target_ori: # if looking at distractor with same orientation as target
                        fix_target_ori += 1
                        
                    if str(distr_color[stim_ind[0]]) == str(target_color): # if looking at distractor with same color as target
                        fix_target_color += 1
                else:
                    #print('Looking at target')
                    fix_target += 1

            # convert to percentage
            nr_distractor_fix = len(trial_fix_df) - fix_target
            if nr_distractor_fix > 0:
                fix_target_color /= nr_distractor_fix # percentage of search fixations, excluding target fixations
                fix_target_ori /= nr_distractor_fix
            else:
                fix_target_color = np.nan
                fix_target_ori = np.nan
            fix_target /= len(trial_fix_df) # percentage of fixations that were on target
        
        else: # if no fixations, then return nan
            fix_target_color = np.nan
            fix_target_ori = np.nan
            fix_target = np.nan

        return fix_target, fix_target_color, fix_target_ori


    def get_fix_on_features_df(self, df_manual_responses = None):

        """ 
        Get percentage of fixations per trial
        that fall on target features
        """

        # make out dir, to save eye events dataframe
        os.makedirs(self.outdir, exist_ok=True)

        # set up empty df
        df_fixations_on_features = pd.DataFrame({'sj': [], 'trial_num': [], 'block_num': [],
                                        'target_ecc': [], 'set_size': [], 
                                        'fix_on_target': [], 'fix_on_target_color': [], 'fix_on_target_ori': []})

        ## loop over participants

        for pp in self.dataObj.sj_num:

            ## get all eye events (fixation and saccades)

            eye_df_filename = op.join(self.outdir, 'sub-{sj}_eye_events.csv'.format(sj = pp))
            
            if not op.isfile(eye_df_filename):
                print('Getting eye events for sub-{sj}'.format(sj = pp))
                eye_events_df = self.get_eyelink_events(self.EYEevents_files['sub-{sj}'.format(sj = pp)], 
                                                        sampling_rate = 1000, 
                                                        save_as = eye_df_filename)
            else:
                print('Loading %s'%eye_df_filename)
                eye_events_df = pd.read_csv(eye_df_filename)


            ## select only correct trials for participant
            pp_manual_response_df = df_manual_responses[(df_manual_responses['correct_response'] == 1) & \
                                                        (df_manual_responses['sj'] == 'sub-{sj}'.format(sj = pp))]

            # loop over ecc
            for e in self.dataObj.ecc:
                
                # loop over set size
                for ss in self.dataObj.set_size:

                    for blk in eye_events_df.block_num.unique().astype(int):
                        # get trial indices for block
                        blk_trials = pp_manual_response_df[(pp_manual_response_df['block_num'] == blk) & \
                                                        (pp_manual_response_df['target_ecc'] == e) & \
                                                        (pp_manual_response_df['set_size'] == ss)].trial_num.values
                
                        for t in blk_trials:

                            fix_target_trl, fix_target_color_trl, fix_target_ori_trl = self.get_trl_percent_fix_on_features(eye_events_df, 
                                                                                                              participant = pp, 
                                                                                                              block_num = blk, 
                                                                                                              trial_num = t,
                                                                                                              hRes = self.dataObj.params['window_extra']['size'][0], 
                                                                                                              vRes = self.dataObj.params['window_extra']['size'][1])

                            # append in dataframe
                            df_fixations_on_features = pd.concat((df_fixations_on_features,
                                                        pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],
                                                                    'trial_num': [t], 
                                                                    'block_num': [blk],
                                                                    'target_ecc': [e], 
                                                                    'set_size': [ss], 
                                                                    'fix_on_target': [fix_target_trl], 
                                                                    'fix_on_target_color': [fix_target_color_trl],
                                                                    'fix_on_target_ori': [fix_target_ori_trl]})), ignore_index = True)

        return df_fixations_on_features

    
    def get_mean_fix_on_features_df(self, df_fixations_on_features = None):

        """ 
        Get mean percentage of fixations 
        that fall on target features for ecc and set size
        """

        df_mean_fix_on_features = pd.DataFrame({'sj': [],'target_ecc': [], 'set_size': [], 
                                                'mean_fix_on_target_color': [], 
                                                'mean_fix_on_target_ori': [],
                                                'mean_fix_on_target': []})

        for pp in df_fixations_on_features.sj.unique():
    
            # loop over ecc
            for e in self.dataObj.ecc:

                # loop over set size
                for ss in self.dataObj.set_size:
                    
                    mean_df = df_fixations_on_features[(df_fixations_on_features['sj'] == pp) & \
                                (df_fixations_on_features['target_ecc'] == e) & \
                                (df_fixations_on_features['set_size'] == ss)].mean()
                    
                    df_mean_fix_on_features = pd.concat((df_mean_fix_on_features,
                                                        pd.DataFrame({'sj': [pp],
                                                                    'target_ecc': [e], 
                                                                    'set_size': [ss], 
                                                                    'mean_fix_on_target_color': [mean_df.fix_on_target_color], 
                                                                    'mean_fix_on_target_ori': [mean_df.fix_on_target_ori],
                                                                    'mean_fix_on_target': [mean_df.fix_on_target]
                                                                    })), ignore_index = True)
        return df_mean_fix_on_features


class EyeTrackCrowding(EyeTrack):

    def __init__(self, dataObj):  # initialize child class

        """ Initializes EyeTrack object. 

        Parameters
        ----------
        dataObj: object
            task behavior object from  load_beh_data.X 
        """

        # need to initialize parent class (EyeTrack), indicating output infos
        super().__init__(dataObj = dataObj)


