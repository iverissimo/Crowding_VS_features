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

    
    def get_search_mean_fixations(self, df_manual_responses = None, exclude_target_fix = True, sampling_rate = 1000, min_fix_start = .150):

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
                                                        sampling_rate = sampling_rate, 
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
                                    # also remove fixations that are too quick
                                    trl_fix_df = trl_fix_df[(trl_fix_df['ev_start_sample'] > (min_fix_start * sampling_rate) + trl_fix_df['phase_start_sample'].values[0])]

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


    def get_search_trl_fixations(self, df_manual_responses = None, exclude_target_fix = True, sampling_rate = 1000, min_fix_start = .150):

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
                                                        sampling_rate = sampling_rate, 
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
                                    # also remove fixations that are too quick
                                    trl_fix_df = trl_fix_df[(trl_fix_df['ev_start_sample'] > (min_fix_start * sampling_rate) + trl_fix_df['phase_start_sample'].values[0])]

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


    def get_fix_slopes(self, df_trl_fixations, fix_nr = True, per_ecc = True):

        """
        calculate search slopes and intercept per ecc

        Parameters
        ----------
        df_manual_responses : DataFrame
            dataframe with results from get_RTs()
        per_ecc: bool
            if we want to get slope per eccentricity or combined over all
        """ 

        # set empty df
        df_search_fix_slopes = pd.DataFrame({'sj': [],'target_ecc': [],  'slope': [], 'intercept': []})

        # loop over subjects
        for pp in self.dataObj.sj_num:

            print('calculating search slopes for sub-{sj}'.format(sj = pp))

            if per_ecc: # loop over ecc
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
            else:
                # sub-select df
                df_temp = df_trl_fixations[(df_trl_fixations['sj'] == 'sub-{sj}'.format(sj = pp))]

                # fit linear regressor
                regressor = LinearRegression()
                if fix_nr:
                    regressor.fit(df_temp[['set_size']], df_temp[['nr_fixations']]) # slope in #fix/item
                else:
                    regressor.fit(df_temp[['set_size']], df_temp[['mean_fix_dur']]*1000) # because we want slope to be in ms/item

                # save df
                df_search_fix_slopes = pd.concat([df_search_fix_slopes, 
                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],  
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

    def get_trl_fixation_feature_selectivity(self, trl_fix_df, pp_trial_info = None, block_num = None, trial_num = None):

        """ 
        For a specific participant trial,
        get the number of fixations that fall on distractors
        with the same color or orientation of target
        and the nr of fixations on target itself
        """

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

        ## append distractor + target position 
        all_stim_pos = distr_pos + [target_pos]

        # kd-tree for quick nearest-neighbor lookup
        tree = spatial.KDTree(all_stim_pos)
        
        # dict with relevant info 
        fix_selectivity = {'target': {'nr': 0, 'mean_duration': [], 'mean_distance': []},
                        'distractor_Tcolor':{'nr': 0, 'mean_duration': [], 'mean_distance': []},
                        'distractor_other':{'nr': 0, 'mean_duration': [], 'mean_distance': []},
                        'distractor_Torientation': {'nr': 0, 'mean_duration': [], 'mean_distance': []}}

        # if there were fixations
        if not trl_fix_df.empty:
            for _,row in trl_fix_df.iterrows(): # iterate over dataframe

                # make positions compatible with display
                fx = row['x_pos'] - self.hRes/2 # start x pos
                fy = row['y_pos'] - self.vRes/2; fy = -fy #start y pos

                # get nearest stim distance and index
                dist, stim_ind = tree.query([(fx,fy)])
                
                # if closest item is not target
                if stim_ind[0] < len(distr_pos): 
                    
                    # if looking at distractor with same orientation as target
                    if distr_ori[stim_ind[0]] == target_ori: 
                        fix_selectivity['distractor_Torientation']['nr'] += 1
                        fix_selectivity['distractor_Torientation']['mean_duration'].append(row['dur_in_sec'])
                        fix_selectivity['distractor_Torientation']['mean_distance'].append(dist)
                    
                    # if looking at distractor with same color as target
                    if str(distr_color[stim_ind[0]]) == str(target_color): 
                        fix_selectivity['distractor_Tcolor']['nr'] += 1
                        fix_selectivity['distractor_Tcolor']['mean_duration'].append(row['dur_in_sec'])
                        fix_selectivity['distractor_Tcolor']['mean_distance'].append(dist)

                    #if looking at distractor with completely different features 
                    if ((distr_ori[stim_ind[0]] != target_ori) and (str(distr_color[stim_ind[0]]) != str(target_color))):
                        fix_selectivity['distractor_other']['nr'] += 1
                        fix_selectivity['distractor_other']['mean_duration'].append(row['dur_in_sec'])
                        fix_selectivity['distractor_other']['mean_distance'].append(dist)
                
                # fixation on target
                else:
                    fix_selectivity['target']['nr'] += 1
                    fix_selectivity['target']['mean_duration'].append(row['dur_in_sec'])
                    fix_selectivity['target']['mean_distance'].append(dist)

            # average duration and distance
            fix_selectivity['distractor_Torientation']['mean_duration'] = np.mean(fix_selectivity['distractor_Torientation']['mean_duration'])
            fix_selectivity['distractor_Torientation']['mean_distance'] = np.mean(fix_selectivity['distractor_Torientation']['mean_distance'])

            fix_selectivity['distractor_Tcolor']['mean_duration'] = np.mean(fix_selectivity['distractor_Tcolor']['mean_duration'])
            fix_selectivity['distractor_Tcolor']['mean_distance'] = np.mean(fix_selectivity['distractor_Tcolor']['mean_distance'])

            fix_selectivity['target']['mean_duration'] = np.mean(fix_selectivity['target']['mean_duration'])
            fix_selectivity['target']['mean_distance'] = np.mean(fix_selectivity['target']['mean_distance'])

            fix_selectivity['distractor_other']['mean_duration'] = np.mean(fix_selectivity['distractor_other']['mean_duration'])
            fix_selectivity['distractor_other']['mean_distance'] = np.mean(fix_selectivity['distractor_other']['mean_distance'])

        else: # if no fixations, then return empty
            fix_selectivity = {}

        return fix_selectivity

    def get_fix_on_features_df(self, df_manual_responses = None, exclude_target_fix = True, sampling_rate = 1000, min_fix_start = .150):

        """ 
        Get percentage of fixations per trial
        that fall on target features
        """

        # make out dir, to save eye events dataframe
        os.makedirs(self.outdir, exist_ok=True)

        # set up empty df
        df_fixations_on_features = pd.DataFrame({'sj': [], 'trial_num': [], 'block_num': [],
                                        'target_ecc': [], 'set_size': [], 'nr_fixations': [], 
                                    'nr_fix_on_T': [], 'mean_fix_dur_on_T': [], 'mean_fix_dist2T': [],
                                    'nr_fix_on_DDF': [], 'mean_fix_dur_on_DDF': [], 'mean_fix_dist2DDF': [],
                                    'nr_fix_on_DTC': [], 'mean_fix_dur_on_DTC': [], 'mean_fix_dist2DTC': [],
                                    'nr_fix_on_DTO': [], 'mean_fix_dur_on_DTO': [], 'mean_fix_dist2DTO': []})

        ## loop over participants
        for pp in self.dataObj.sj_num:

            ## get all eye events (fixation and saccades)
            eye_df_filename = op.join(self.outdir, 'sub-{sj}_eye_events.csv'.format(sj = pp))
            
            if not op.isfile(eye_df_filename):
                print('Getting eye events for sub-{sj}'.format(sj = pp))
                eye_events_df = self.get_eyelink_events(self.EYEevents_files['sub-{sj}'.format(sj = pp)], 
                                                        sampling_rate = sampling_rate, 
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

                            # if fixations on trial and we want to exclude fixations on target
                            if not trl_fix_df.empty and exclude_target_fix: # if 
                                    
                                ## get target coordinates
                                target_pos = pp_trial_info[(pp_trial_info['block'] == blk) & \
                                                        (pp_trial_info['index'] == t)].target_pos.values[0]

                                # get x,y coordinates of last fixation
                                fix_x = trl_fix_df.iloc[-1]['x_pos'] - self.hRes/2 # start x pos
                                fix_y = trl_fix_df.iloc[-1]['y_pos'] - self.vRes/2; fix_y = -fix_y #start y pos
                                
                                # if last fixation on target, remove
                                if ((target_pos[0] - fix_x)**2 + (target_pos[-1] - fix_y)**2) <= self.r_gabor ** 2:
                                    trl_fix_df = trl_fix_df.iloc[:-1] 

                            # if fixations on trial  
                            if not trl_fix_df.empty:
                                # also remove fixations that are too quick
                                trl_fix_df = trl_fix_df[(trl_fix_df['ev_start_sample'] > (min_fix_start * sampling_rate) + trl_fix_df['phase_start_sample'].values[0])]

                                # check where trial fixations fall on - call function
                                fix_selectivity = self.get_trl_fixation_feature_selectivity(trl_fix_df, 
                                                                            pp_trial_info = pp_trial_info, 
                                                                            block_num = blk, trial_num = t)

                                if len(fix_selectivity) > 0: # if not empty

                                    df_fixations_on_features = pd.concat((df_fixations_on_features,
                                                        pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],
                                                                    'trial_num': [t], 
                                                                    'block_num': [blk],
                                                                    'target_ecc': [e], 
                                                                    'set_size': [ss], 
                                                                    'nr_fixations': [fix_selectivity['target']['nr'] + fix_selectivity['distractor_Tcolor']['nr'] + \
                                                                                    fix_selectivity['distractor_Torientation']['nr'] + fix_selectivity['distractor_other']['nr']],
                                                                    'nr_fix_on_T': [fix_selectivity['target']['nr']], 
                                                                    'mean_fix_dur_on_T': [fix_selectivity['target']['mean_duration']],
                                                                    'mean_fix_dist2T': [fix_selectivity['target']['mean_distance']],
                                                                    'nr_fix_on_DTC': [fix_selectivity['distractor_Tcolor']['nr']], 
                                                                    'mean_fix_dur_on_DTC': [fix_selectivity['distractor_Tcolor']['mean_duration']], 
                                                                    'mean_fix_dist2DTC': [fix_selectivity['distractor_Tcolor']['mean_distance']],
                                                                    'nr_fix_on_DTO': [fix_selectivity['distractor_Torientation']['nr']], 
                                                                    'mean_fix_dur_on_DTO': [fix_selectivity['distractor_Torientation']['mean_duration']], 
                                                                    'mean_fix_dist2DTO': [fix_selectivity['distractor_Torientation']['mean_distance']],
                                                                    'nr_fix_on_DDF': [fix_selectivity['distractor_other']['nr']], 
                                                                    'mean_fix_dur_on_DDF': [fix_selectivity['distractor_other']['mean_duration']], 
                                                                    'mean_fix_dist2DDF': [fix_selectivity['distractor_other']['mean_distance']]})
                                                                    ), ignore_index = True)

        return df_fixations_on_features

    def get_scanpath_ratio_df(self, df_manual_responses = None, sampling_rate = 1000, min_fix_start = .150):

        """ 
        Get percentage of fixations per trial
        that fall on target features
        """

        # make out dir, to save eye events dataframe
        os.makedirs(self.outdir, exist_ok=True)

        # set up empty df
        df_scanpath_ratio = pd.DataFrame({'sj': [], 'trial_num': [], 'block_num': [],
                                        'target_ecc': [], 'set_size': [], 'nr_fixations': [], 
                                        'sum_euclidean_dist': [], 'ratio': []})

        ## loop over participants

        for pp in self.dataObj.sj_num:

            ## get all eye events (fixation and saccades)

            eye_df_filename = op.join(self.outdir, 'sub-{sj}_eye_events.csv'.format(sj = pp))
            
            if not op.isfile(eye_df_filename):
                print('Getting eye events for sub-{sj}'.format(sj = pp))
                eye_events_df = self.get_eyelink_events(self.EYEevents_files['sub-{sj}'.format(sj = pp)], 
                                                        sampling_rate = sampling_rate, 
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

                            # if fixations on trial  
                            if not trl_fix_df.empty:
                                # also remove fixations that are too quick
                                trl_fix_df = trl_fix_df[(trl_fix_df['ev_start_sample'] > (min_fix_start * sampling_rate) + trl_fix_df['phase_start_sample'].values[0])]

                                if not trl_fix_df.empty:
                                    ## get target coordinates
                                    target_pos = pp_trial_info[(pp_trial_info['block'] == blk) & \
                                                            (pp_trial_info['index'] == t)].target_pos.values[0]

                                    # get target distance from initial fixation in pixels
                                    # (Euclidean distance between the fixation cross location and the target)
                                    target_dist_pix = np.sqrt((target_pos[0] - 0)**2 + (target_pos[-1] - 0)**2)

                                    # now get summed Euclidean distances of the eye movements made while
                                    # searching for the target
                                    fix_x = trl_fix_df.iloc[0]['x_pos'] - self.hRes/2 # start x pos
                                    fix_y = trl_fix_df.iloc[0]['y_pos'] - self.vRes/2; fix_y = -fix_y #start y pos

                                    sum_dist_pix = np.sqrt((fix_x - 0)**2 + (fix_y - 0)**2)

                                    # if more than one fixation made
                                    if len(trl_fix_df) > 1:
                                        for f_ind in np.arange(len(trl_fix_df)-1):

                                            fix_x0 = trl_fix_df.iloc[f_ind]['x_pos'] - self.hRes/2 # start x pos
                                            fix_y0 = trl_fix_df.iloc[f_ind]['y_pos'] - self.vRes/2; fix_y0 = -fix_y0 #start y pos

                                            fix_x1 = trl_fix_df.iloc[f_ind+1]['x_pos'] - self.hRes/2 # start x pos
                                            fix_y1 = trl_fix_df.iloc[f_ind+1]['y_pos'] - self.vRes/2; fix_y1 = -fix_y1 #start y pos

                                            sum_dist_pix += np.sqrt((fix_x0 - fix_x1)**2 + (fix_y0 - fix_y1)**2)

                                    # concatenate
                                    df_scanpath_ratio = pd.concat((df_scanpath_ratio,
                                                                pd.DataFrame({'sj': ['sub-{sj}'.format(sj = pp)],
                                                                        'trial_num': [t], 
                                                                        'block_num': [blk],
                                                                        'target_ecc': [e], 
                                                                        'set_size': [ss],  
                                                                        'nr_fixations': [len(trl_fix_df)], 
                                                                        'sum_euclidean_dist': [sum_dist_pix], 
                                                                        'ratio': [sum_dist_pix/target_dist_pix]})), ignore_index = True)
                            
        return df_scanpath_ratio

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

    def get_trl_feature_selective_ranked_fixations(self, trl_fix_df, pp_trial_info = None, block_num = None, trial_num = None):

        """ 
        For a specific participant trial,
        make df will all fixations that fall on DISTRACTORS
        with the same color or orientation of target
        ranked by order of appearance
        """

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

        ## append distractor + target position 
        all_stim_pos = distr_pos + [target_pos]

        # kd-tree for quick nearest-neighbor lookup
        tree = spatial.KDTree(all_stim_pos)
        
        # dict with relevant info 
        fix_selectivity = {'fixation_object': [], 'rank': [], 'distance': [], 'duration': []}

        counter = 0
        # if there were fixations
        if not trl_fix_df.empty:
            for _,row in trl_fix_df.iterrows(): # iterate over dataframe

                # make positions compatible with display
                fx = row['x_pos'] - self.hRes/2 # start x pos
                fy = row['y_pos'] - self.vRes/2; fy = -fy #start y pos

                # get nearest stim distance and index
                dist, stim_ind = tree.query([(fx,fy)])
                
                # if closest item is not target
                if stim_ind[0] < len(distr_pos): 

                    # if looking at distractor with same orientation as target
                    if distr_ori[stim_ind[0]] == target_ori: 
                        fix_selectivity['fixation_object'].append('DTO')
                        
                    # if looking at distractor with same color as target
                    if str(distr_color[stim_ind[0]]) == str(target_color): 
                        fix_selectivity['fixation_object'].append('DTC')

                    #if looking at distractor with completely different features 
                    if ((distr_ori[stim_ind[0]] != target_ori) and (str(distr_color[stim_ind[0]]) != str(target_color))):
                        fix_selectivity['fixation_object'].append('DDF')

                # fixation on target
                else:
                    fix_selectivity['fixation_object'].append('T')

                fix_selectivity['rank'].append(counter)
                fix_selectivity['distance'].append(dist[0])
                fix_selectivity['duration'].append(row['dur_in_sec'])

                counter += 1
    
        else: # if no fixations, then return empty
            fix_selectivity = {}

        return fix_selectivity

    def get_ALLfix_on_features_df(self, df_manual_responses = None, exclude_target_fix = True, sampling_rate = 1000, min_fix_start = .150):

        """ 
        Make df that stores all fixations,
        variation of self.get_fix_on_features_df
        but now without averaging duration and distance of fixation per trial
        """

        # make out dir, to save eye events dataframe
        os.makedirs(self.outdir, exist_ok=True)

        # set up empty df
        df_fixations_on_features = pd.DataFrame({'sj': [], 'trial_num': [], 'block_num': [],
                                        'target_ecc': [], 'set_size': [], 'fixation_object': [], 
                                        'trial_rank': [], 'distance': [], 'duration': []})

        ## loop over participants

        for pp in self.dataObj.sj_num:

            ## get all eye events (fixation and saccades)
            eye_df_filename = op.join(self.outdir, 'sub-{sj}_eye_events.csv'.format(sj = pp))
            
            if not op.isfile(eye_df_filename):
                print('Getting eye events for sub-{sj}'.format(sj = pp))
                eye_events_df = self.get_eyelink_events(self.EYEevents_files['sub-{sj}'.format(sj = pp)], 
                                                        sampling_rate = sampling_rate, 
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

                            # if fixations on trial and we want to exclude fixations on target
                            if not trl_fix_df.empty and exclude_target_fix: # if 
                                    
                                ## get target coordinates
                                target_pos = pp_trial_info[(pp_trial_info['block'] == blk) & \
                                                        (pp_trial_info['index'] == t)].target_pos.values[0]

                                # get x,y coordinates of last fixation
                                fix_x = trl_fix_df.iloc[-1]['x_pos'] - self.hRes/2 # start x pos
                                fix_y = trl_fix_df.iloc[-1]['y_pos'] - self.vRes/2; fix_y = -fix_y #start y pos
                                
                                # if last fixation on target, remove
                                if ((target_pos[0] - fix_x)**2 + (target_pos[-1] - fix_y)**2) <= self.r_gabor ** 2:
                                    trl_fix_df = trl_fix_df.iloc[:-1] 

                            # if fixations on trial  
                            if not trl_fix_df.empty:
                                # also remove fixations that are too quick
                                trl_fix_df = trl_fix_df[(trl_fix_df['ev_start_sample'] > (min_fix_start * sampling_rate) + trl_fix_df['phase_start_sample'].values[0])]

                                # check where trial fixations fall on - call function
                                fix_selectivity = self.get_trl_feature_selective_ranked_fixations(trl_fix_df, 
                                                                            pp_trial_info = pp_trial_info, 
                                                                            block_num = blk, trial_num = t)

                                if len(fix_selectivity) > 0: # if not empty
                                    arr_len = len(fix_selectivity['rank'])

                                    df_fixations_on_features = pd.concat((df_fixations_on_features,
                                                        pd.DataFrame({'sj': np.tile('sub-{sj}'.format(sj = pp), arr_len),
                                                                    'trial_num': np.tile(t, arr_len), 
                                                                    'block_num': np.tile(blk, arr_len),
                                                                    'target_ecc': np.tile(e, arr_len), 
                                                                    'set_size': np.tile(ss, arr_len), 
                                                                    'fixation_object': fix_selectivity['fixation_object'],
                                                                    'trial_rank': fix_selectivity['rank'],
                                                                    'distance': fix_selectivity['distance'],
                                                                    'duration': fix_selectivity['duration']})
                                                                    ), ignore_index = True)

        return df_fixations_on_features


    def get_OriVSdata_fixations(self, ecc = [4, 8, 12], setsize = [5,15,30], minRT = .250, max_RT = 5, 
                                prev_vRes = 1050, prev_hRes=1680):

        """
        Helper function to count number of fixations for search data
        of previous experiment. Based on what was done in other repo
        """
        # append Pygaz analyser folder, cloned from https://github.com/esdalmaijer/PyGazeAnalyser.git
        sys.path.append(op.join('/Users/verissimo/Documents/Projects/PSR_2019/Crowding','PyGazeAnalyser'))
        from pygazeanalyser.edfreader import read_edf

        ## Load summary measures
        previous_dataset_pth = self.dataObj.params['paths']['data_ori_vs_pth']
        sum_measures = np.load(op.join(previous_dataset_pth,
                                    'summary','sum_measures.npz')) # all relevant measures

        ## get list with all participants
        # that passed the inclusion criteria
        previous_sub_list = [val for val in sum_measures['all_subs'] if val not in sum_measures['excluded_sub']]

        # if fixations between 150ms after display and key press time
        sample_thresh = 1000 * 0.150 # 1000Hz * time in seconds

        # gabor radius in pixels
        prev_r_gabor = (2.2/2)/self.get_dva_per_pix(height_cm = 30, 
                                                    distance_cm = 57, 
                                                    vert_res_pix = prev_vRes)

        # set up empty df
        df_fixations = pd.DataFrame({'sj': [], 'trial': [], 'target_ecc': [], 'set_size': [], 'nr_fixations': []})

        # loop over participants
        for prev_pp in previous_sub_list:

            # load behav data search
            prev_pp_VSfile = op.join(previous_dataset_pth, 'output_VS', 
                                        'data_visualsearch_pp_{p}.csv'.format(p = prev_pp))
            df_vs = pd.read_csv(prev_pp_VSfile, sep='\t')

            # load eye data
            prev_pp_eye_VSfile = op.join(previous_dataset_pth, 'output_VS', 
                               'eyedata_visualsearch_pp_{p}.asc'.format(p = prev_pp))
            eyedata_vs = read_edf(prev_pp_eye_VSfile, 'start_trial', stop='stop_trial', debug=False)

            # sub select behavioral dataframe 
            df_tmp = df_vs[(df_vs['key_pressed'] == df_vs['target_orientation']) & \
                            (df_vs['RT'] > minRT) & \
                            (df_vs['RT'] < max_RT)]

            # loop over trials
            for ind, trl_num in enumerate(df_tmp.index.values):

                # index for moment when display was shown
                idx_display = np.where(np.array(eyedata_vs[trl_num]['events']['msg'])[:,-1]=='var display True\n')[0][0]
                # eye tracker sample time of display
                smp_display = eyedata_vs[trl_num]['events']['msg'][idx_display][0]             

                # get list of fixations in trial
                trl_fix_list = [arr for _,arr in enumerate(eyedata_vs[trl_num]['events']['Efix']) if (arr[0] > smp_display+sample_thresh) and \
                                        (arr[0] < np.round(smp_display + df_tmp.RT.values[ind]*1000))]

                # if there are fixations in trial
                if len(trl_fix_list) > 0:
                    # get target position as strings in list
                    target_pos = df_tmp.target_position.values[ind].replace(']','').replace('[','').split(' ')
                    # convert to list of floats
                    target_pos = np.array([float(val) for i,val in enumerate(target_pos) if len(val)>1])

                    ## check if last fixation on target
                    fix_x = trl_fix_list[-1][-2] - prev_hRes/2
                    fix_y = trl_fix_list[-1][-1] - prev_vRes/2; fix_y = - fix_y

                    if np.sqrt((fix_x-target_pos[0])**2+(fix_y-target_pos[1])**2) <= prev_r_gabor: # if it was, then remove
                        trl_fix_list = trl_fix_list[:-1]

                ## concatenate in DataFrame
                df_fixations = pd.concat((df_fixations,
                                        pd.DataFrame({'sj': prev_pp, 
                                        'trial': [trl_num], 
                                        'target_ecc': [df_tmp.target_ecc.values[ind]], 
                                        'set_size': [df_tmp.set_size.values[ind]], 
                                        'nr_fixations': [len(trl_fix_list)]})
                                        ), ignore_index = True)
                
        return df_fixations

    def get_OriVSdata_FixSlopes(self, df_previous_fixations):

        """
        Helper function to calculate RT slopes for search data
        of previous experiment. 
        Expects fixation dataframe from self.get_OriVSdata_fixations
        """

        # set up empty df
        df_slope_results = pd.DataFrame({'sj': [], 'slope': [], 'intercept': []})

        ## loop over participants
        for prev_pp in df_previous_fixations.sj.unique():

            df_tmp = df_previous_fixations[df_previous_fixations['sj'] == prev_pp]

            # fit linear regressor
            regressor = LinearRegression()
            regressor.fit(df_tmp[['set_size']], df_tmp[['nr_fixations']])

            # save df
            df_slope_results = pd.concat((df_slope_results, 
                                        pd.DataFrame({'sj': [prev_pp],
                                                    'slope': [regressor.coef_[0][0]],
                                                    'intercept': [regressor.intercept_[0]]})), ignore_index = True)

        return df_slope_results






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


