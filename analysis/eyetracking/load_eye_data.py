import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml

from analysis.behaviour import load_beh_data
from scipy.signal import savgol_filter
import utils


class EyeTrackCrowding(load_beh_data.BehCrowding):
    
    def __init__(self, params, sj_num, session, exclude_sj = []):  # initialize child class

        """ Initializes EyeTrackCrowding object. 
      
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, session = session, exclude_sj = exclude_sj)
        
        
class EyeTrackVsearch(load_beh_data.BehVsearch):
   
    def __init__(self, params, sj_num, session, exclude_sj = []):  # initialize child class

        """ Initializes EyeTrackVsearch object. 
      
        Parameters
        ----------
        params : str/yaml dict
            where parameters from experiment and analysis live
        sj_num : str
            participant number
        session : str
            session type (test/train)
        """

        # need to initialize parent class (BehTask), indicating output infos
        super().__init__(params = params, sj_num = sj_num, session = session, exclude_sj = exclude_sj) 
        
    
    def convert2asc(self, task_name, replace_val = 0.0001):
        
        """
        Convert edf files to asc samples and message files
        """
        
        EYEsamples_files = {}
        EYEevents_files = {}

        for i, pp in enumerate(self.sj_num):

            # subject data folder
            sj_pth = op.join(op.join(self.data_path, 'sourcedata', 'sub-{sj}'.format(sj = pp)))

            # load trial info, with step up for that session
            edf_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.session)) and \
                            'task-{task}'.format(task = task_name) in x and x.endswith('.edf')]
            
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
        
    
    def get_raw_eyetrackdata(self):
        
        """
        Load asc samples and message files save dataframe with relevant RAW sample data
        (this is, for each trial, not preprocessed)
        """
        
        df_raw_data = {}
        
        for i, pp in enumerate(self.sj_num):
            
            print('Loading raw data for sub-{sj}'.format(sj = pp))

            ## read eyetracking events file as np.array, 
            # to get relevant structure
            #
            f = open(self.EYEevents_files['sub-{sj}'.format(sj = pp)],'r')
            fileTxt0 = np.array(list(filter(None, f.read().splitlines(True)))) # split into lines and remove emptys
            f.close()

            # get indices with messages indicating trial starts
            # and select those lines
            trl_msg = [val.replace('\n', '') for val in fileTxt0 if 'start_type-stim_trial' in val and 'MSG']

            # transform into dataframe, for ease of use
            df_start_trial = pd.DataFrame([s.split("\t") for s in trl_msg])
            df_start_trial = df_start_trial[[1,2]]
            df_start_trial.columns = ['sample', 'event_name'] # set column names
            df_start_trial['sample'] = df_start_trial['sample'].astype(float) # set samples to float

            ## STIM start time for each block 
            blk_start_samples = df_start_trial.loc[df_start_trial['event_name'] == 'start_type-stim_trial-0_phase-2']['sample'].values

            ## now make data frame with start trial samples
            # for each trial phase (stim and iti)

            # total number of trials
            nr_trials = np.nanmax(self.trial_info_df['index'].values) # number of trials in block

            # start empty df for subject key
            ev_samples_df = pd.DataFrame({'block_num': [], 'trial_num': [], 
                                            'phase_name': [], 'sample': []})

            for blk, blk_sample in enumerate(blk_start_samples):

                # subselect for block
                blk_df = df_start_trial[(df_start_trial['sample'] >= blk_start_samples[blk])].drop_duplicates(subset = 'event_name', 
                                                                                                           keep = 'first')

                ## set stim and iti samples in dict, 
                # for first trial (has different phase than rest)
                ev_samples_df = pd.concat([ev_samples_df,
                                           pd.DataFrame({'block_num': [blk], 'trial_num': [0], 'phase_name': ['stim'], 
                                 'sample': blk_df.loc[blk_df['event_name'] == 'start_type-stim_trial-0_phase-2']['sample'].values})])

                ev_samples_df = pd.concat([ev_samples_df,
                                           pd.DataFrame({'block_num': [blk], 'trial_num': [0], 'phase_name': ['iti'], 
                                 'sample': blk_df.loc[blk_df['event_name'] == 'start_type-stim_trial-0_phase-3']['sample'].values})])

                # then add rest of trials
                for t in np.arange(nr_trials):   
                    ev_samples_df = pd.concat([ev_samples_df,
                                           pd.DataFrame({'block_num': [blk], 'trial_num': [t+1], 'phase_name': ['stim'], 
                            'sample': blk_df.loc[blk_df['event_name'] == 'start_type-stim_trial-{t}_phase-0'.format(t=t+1)]['sample'].values})])

                    ev_samples_df = pd.concat([ev_samples_df,
                                           pd.DataFrame({'block_num': [blk], 'trial_num': [t+1], 'phase_name': ['iti'], 
                            'sample': blk_df.loc[blk_df['event_name'] == 'start_type-stim_trial-{t}_phase-1'.format(t=t+1)]['sample'].values})])


            ## load raw gaze data into data frame
            # has all data from start to finish recording
            df_eyelink_data = pd.read_csv(data_search.EYEsamples_files['sub-{sj}'.format(sj = pp)], sep='\t', 
                                         header=None, usecols=np.arange(6),
                                        names = ['sample', 'L_gaze_x', 'L_gaze_y', 'L_pupil', 'L_vel_x', 'L_vel_y'])
            
            ## organize it into trial data frame
            # because we dont care for stuff outside "stim on screen" time + ITI (to account for saccade landing after key press)

            df_raw_data['sub-{sj}'.format(sj = pp)] = pd.DataFrame({'block_num': [], 'trial_num': [], 'sample': [], 
                                                                    'gaze_x': [], 'gaze_y': [], 'pupil': [], 
                                                                    'vel_x': [], 'vel_y': []})

            # loop over blocks
            for blk in ev_samples_df['block_num'].unique():

                # and trials
                for trl in ev_samples_df['trial_num'].unique():

                    trl_start_df = ev_samples_df.loc[(ev_samples_df['block_num'] == blk) & (ev_samples_df['trial_num'] == trl)]

                    trl_eyelink_df = df_eyelink_data.loc[(df_eyelink_data['sample'] >= trl_start_df[trl_start_df['phase_name']=='stim']['sample'].values[0]) & \
                                       (df_eyelink_data['sample'] <= trl_start_df[trl_start_df['phase_name']=='iti']['sample'].values[-1])]

                    # append
                    df_raw_data['sub-{sj}'.format(sj = pp)] = pd.concat([df_raw_data['sub-{sj}'.format(sj = pp)],
                                                        pd.DataFrame({'block_num': np.tile(blk, len(trl_eyelink_df)), 
                                                          'trial_num': np.tile(trl, len(trl_eyelink_df)), 
                                                          'sample': trl_eyelink_df['sample'].values, 
                                                          'gaze_x': trl_eyelink_df['L_gaze_x'].values, 
                                                          'gaze_y': trl_eyelink_df['L_gaze_y'].values, 
                                                          'pupil': trl_eyelink_df['L_pupil'].values, 
                                                          'vel_x': trl_eyelink_df['L_vel_x'].values, 
                                                          'vel_y': trl_eyelink_df['L_vel_y'].values})])
        
        # save
        self.df_raw_data = df_raw_data
        
        
    def get_proc_eyetrackdata(self, sampl_freq = 1000, sg_order = 2, sg_window_length = 21):
        
        """
        Convert raw data frame of gaze to preprocessed data frame
        also calculates velocity and acceleration
        """
        
        df_proc_data = {}
        
        for i, pp in enumerate(self.sj_num):
            
            print('Processing eye data for sub-{sj}'.format(sj = pp))
            
            ## process data 

            df_proc_data['sub-{sj}'.format(sj = pp)] = pd.DataFrame({'block_num': [], 'trial_num': [], 'sample': [], 
                                                                    'gaze_x': [], 'gaze_y': [], 'pupil': [], 
                                                                    'velocity': [], 'acceleration': []})

            # loop over blocks
            for blk in self.df_raw_data['sub-{sj}'.format(sj = pp)]['block_num'].unique():

                # and trials
                for trl in self.df_raw_data['sub-{sj}'.format(sj = pp)]['trial_num'].unique():

                    trl_start_df = self.df_raw_data['sub-{sj}'.format(sj = pp)].loc[(self.df_raw_data['sub-{sj}'.format(sj = pp)]['block_num'] == blk) & \
                                                                                    (self.df_raw_data['sub-{sj}'.format(sj = pp)]['trial_num'] == trl)]

                    # filter X and Y
                    filtered_x = savgol_filter(trl_start_df['gaze_x'].values, int(sg_window_length * 1000/sampl_freq), sg_order) # 20 ms, 2nd order polynomial 
                    filtered_y = savgol_filter(trl_start_df['gaze_y'].values, int(sg_window_length * 1000/sampl_freq), sg_order) # 20 ms, 2nd order polynomial 

                    ## calculate velocity
                    # in pix/s
                    vel = (np.sqrt(np.diff(filtered_x)**2 + np.diff(filtered_y)**2))/np.diff(trl_start_df['sample'].values) * sampl_freq

                    ## calculate acceleration
                    # in pix/sˆ2
                    acc = vel/np.diff(trl_start_df['sample'].values) * sampl_freq


                    # append
                    df_proc_data['sub-{sj}'.format(sj = pp)] = pd.concat([df_proc_data['sub-{sj}'.format(sj = pp)],
                                                                    pd.DataFrame({'block_num': np.tile(blk, len(vel)), 
                                                                      'trial_num': np.tile(trl, len(vel)), 
                                                                      'sample': trl_start_df['sample'].values[:-1], 
                                                                      'gaze_x': filtered_x[:-1], 
                                                                      'gaze_y': filtered_y[:-1], 
                                                                      'pupil': trl_start_df['pupil'].values[:-1], 
                                                                      'velocity': vel, 
                                                                      'acceleration': acc})])

        # save
        self.df_proc_data = df_proc_data
        
        
    def get_saccades(self, sampl_freq = 1000, init_thresh = 150, std_vel_margin = 6, min_sacc_dur = 10,
                    min_fix_dur = 40, offset_alpha = .7, offset_beta = .3, max_sacc_vel = 1000):
        
        """
        Get saccades for each trial
        
        Parameters
        ----------
        sampl_freq: int
            sampling frequency in Hz
        init_thresh : int/float
            initial threshold value to use
        std_vel_margin: int/float
            standard deviation margin
        min_sacc_dur: int/float
            minimum saccade duration in ms
        min_fix_dur: int/float
            minimum fixation duration in ms
        offset_alpha: float
            alpha weight for offset threshold (weight for whole trial velocity noise)
        offset_beta: float
            beta weight for offset threshold (weight for locally adaptive noise factor)
        max_sacc_vel: int/float
            max saccadic velocity in degrees/sec
        
        """
        
        # get degrees per pixel, for conversions
        self.degree_1pix = utils.get_dva_per_pix(height_cm = self.params['monitor_extra']['height'], 
                                                distance_cm = self.params['monitor']['distance'], 
                                                vert_res_pix = self.params['window_extra']['size'][-1])
        
        df_saccades = {}
        
        for i, pp in enumerate(self.sj_num):
            
            print('Getting saccades for sub-{sj}'.format(sj = pp))

            df_saccades['sub-{sj}'.format(sj = pp)] = pd.DataFrame({'block_num': [], 'trial_num': [], 'onset_samp': [], 
                                                                    'offset_samp': [], 'onset_sec': [], 'offset_sec': [], 
                                                                    'duration_sec': [], 'xpos_onset_pix': [], 'ypos_onset_pix': [],
                                                                    'xpos_offset_pix': [], 'ypos_offset_pix': [], 'displacement': [],
                                                                    'peak_vel': [], 'peak_acc': []})

            # loop over blocks
            for blk in self.df_proc_data['sub-{sj}'.format(sj = pp)]['block_num'].unique():

                # and trials
                for trl in self.df_proc_data['sub-{sj}'.format(sj = pp)]['trial_num'].unique():
                    
                    #print(trl)

                    trl_start_df = self.df_proc_data['sub-{sj}'.format(sj = pp)].loc[(self.df_proc_data['sub-{sj}'.format(sj = pp)]['block_num'] == blk) & \
                                                                                    (self.df_proc_data['sub-{sj}'.format(sj = pp)]['trial_num'] == trl)]
                    
                    # convert velocity from pix/s to deg/sec
                    vel_deg = trl_start_df['velocity'].values * self.degree_1pix
                    samples = trl_start_df['sample'].values
                    
                    # get peak velocity threshold for trial
                    peak_thresh = utils.get_peak_vel_threshold(vel_deg, samples, 
                                                         init_thresh = init_thresh, std_vel_margin = std_vel_margin)
                    
                    # get saccade onsets and offsets, from initial selection
                    curr_sacc = utils.get_initial_saccade_onoffset(vel_deg, samples, peak_thresh,
                                                                  min_sacc_dur = min_sacc_dur, sampl_freq = sampl_freq,
                                                                  max_sacc_vel = max_sacc_vel)
                    
                    ## set empty list where we will store final saccade samples [on, off]
                    final_sacc = []

                    ## for each saccade, confirm on/off times
                    #
                    # first set saccade onset threshold
                    onset_thresh = np.zeros(samples.shape)
                    
                    for _,sacc_times in enumerate(curr_sacc):
                        
                        onset_thresh[np.where((samples > sacc_times[0]) & (samples < sacc_times[-1]))[0]] = 1  
                        
                    onset_thresh = np.mean(vel_deg[onset_thresh<1]) + np.std(vel_deg[onset_thresh<1]) * 3

                    # then set saccade offset threshold
                    # only first part of equation (locally adaptive noise calculated for each saccade individually)
                    offset_thresh_A = offset_alpha * onset_thresh

                    # store real onset and offset
                    for _,sacc_times in enumerate(curr_sacc):

                        ## SACCADE ONSET
                        # find the first sample that goes below threshold, going backwards in time
                        #
                        potential_onsets = list(reversed(np.where((samples <= sacc_times[0]))[0])) # indices for potential onset samples (in reversed order)
                        # go through indices 
                        for i,val in enumerate(potential_onsets): 

                            if (i > 0) and (vel_deg[val] < onset_thresh): # if velocity below saccade onset thresh
                                if vel_deg[val] - vel_deg[potential_onsets[i-1]] >= 0: # if vel i - vel i+1 > 0 (first decrease, before increase in vel)
                                    onset_ind = val # save sample
                                    break

                        ## SACCADE OFFSET
                        # calculate saccade offset thresh
                        offset_thresh_B = np.mean(vel_deg[np.where(samples >= samples[onset_ind]-(min_fix_dur*1000)/sampl_freq)[0][0]:onset_ind]) + \
                                            np.std(vel_deg[np.where(samples >= samples[onset_ind]-(min_fix_dur*1000)/sampl_freq)[0][0]:onset_ind]) * 3

                        offset_thresh = offset_thresh_A + offset_thresh_B * offset_beta

                        #
                        potential_offsets = np.where((samples >= sacc_times[-1]))[0]
                        # go through indices 
                        offset_ind = np.nan
                        for i,val in enumerate(potential_offsets): 

                            if (i > 0) and (vel_deg[val] < offset_thresh): # if velocity below saccade offset thresh
                                if vel_deg[potential_offsets[i-1]] - vel_deg[val] <= 0: # if vel i - vel i+1 < 0 (first decrease in vel)
                                    offset_ind = val # save sample
                                    break

                        # if we never reached condition, 
                        # then remove saccade completely (means we don't have info on end point)
                        if not np.isnan(offset_ind):
                            # append to final sample array
                            final_sacc.append([samples[onset_ind], samples[offset_ind]])
                        
                    ## reunite saccade values of interest
                    onset_samp = []
                    offset_samp = []
                    xpos_onset_pix = []
                    ypos_onset_pix = []
                    xpos_offset_pix = []
                    ypos_offset_pix = []
                    displacement = []
                    peak_vel = []
                    peak_acc = []
                    
                    trial_onset = trl_start_df['sample'].values[0]
                    
                    for ind_s, sacc in enumerate(final_sacc):
                        
                        # temporary df
                        tmp_df = trl_start_df.loc[(trl_start_df['sample'] >= sacc[0]) & (trl_start_df['sample'] <= sacc[-1])]
                        
                        # sample on/offset
                        onset_samp.append(sacc[0])
                        offset_samp.append(sacc[-1])
                        
                        # start position pixel
                        xpos_onset_pix.append(tmp_df['gaze_x'].values[0])
                        ypos_onset_pix.append(tmp_df['gaze_y'].values[0])
                        
                        # end position pixel
                        xpos_offset_pix.append(tmp_df['gaze_x'].values[-1])
                        ypos_offset_pix.append(tmp_df['gaze_y'].values[-1])
                        
                        # total displacement (length) in pix
                        displacement.append(np.sqrt((tmp_df['gaze_x'].values[-1] - tmp_df['gaze_x'].values[0])**2 + (tmp_df['gaze_y'].values[-1] - tmp_df['gaze_y'].values[0])**2))
                        
                        # peak velocity and acceleration
                        peak_vel.append(np.max(tmp_df['velocity'].values))
                        peak_acc.append(np.max(tmp_df['acceleration'].values))
                        
                    ## now store in data frame
                    df_saccades['sub-{sj}'.format(sj = pp)] = pd.concat([df_saccades['sub-{sj}'.format(sj = pp)],
                                                                   pd.DataFrame({'block_num': np.tile(blk, len(final_sacc)), 
                                                                                 'trial_num': np.tile(trl, len(final_sacc)),
                                                                                 'onset_samp': onset_samp, 
                                                                                 'offset_samp': offset_samp, 
                                                                                 'onset_sec': (np.array(onset_samp) - trial_onset)/1000, 
                                                                                 'offset_sec': (np.array(offset_samp) - trial_onset)/1000, 
                                                                                 'duration_sec': (np.array(offset_samp) - np.array(onset_samp))/1000, 
                                                                                 'xpos_onset_pix': xpos_onset_pix, 
                                                                                 'ypos_onset_pix': ypos_onset_pix,
                                                                                 'xpos_offset_pix': xpos_offset_pix, 
                                                                                 'ypos_offset_pix': ypos_offset_pix, 
                                                                                 'displacement': displacement,
                                                                                 'peak_vel': peak_vel, 
                                                                                 'peak_acc': peak_acc})])
                    
        self.df_saccades = df_saccades
        
