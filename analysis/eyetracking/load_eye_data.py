import numpy as np
import os, sys
import os.path as op
import pandas as pd
import yaml

from analysis.behaviour import load_beh_data
from scipy.signal import savgol_filter
import utils

import re

class EyeTrack:

    def __init__(self, behObj):  # initialize child class

        """ Initializes EyeTrack object. 

        Parameters
        ----------
        behObj: object
            task behavior object from  load_beh_data.X 
        """

        self.behObj = behObj
        self.outdir = op.join(self.behObj.derivatives_pth, 'eyetracking')


    def convert2asc(self, replace_val = 0.0001):
        
        """
        Convert edf files to asc samples and message files
        """
        
        EYEsamples_files = {}
        EYEevents_files = {}

        for i, pp in enumerate(self.behObj.sj_num):

            # subject data folder
            sj_pth = op.join(op.join(self.behObj.sourcedata_pth, 'sub-{sj}'.format(sj = pp)))

            # load trial info, with step up for that session
            edf_files = [op.join(sj_pth,x) for x in os.listdir(sj_pth) if x.startswith('sub-{sj}_ses-{ses}'.format(sj = pp, 
                                                                                                                    ses = self.behObj.session)) and \
                            'task-{task}'.format(task = self.behObj.task_name) in x and x.endswith('.edf')]
            
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

    
    def get_eyelink_events(self, filename, sampling_rate = 1000):
        
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

                    if float(ev_list[2]) >= phase_start_sample: # if event happens after trial phase started

                        ## fixations
                        if 'FIX' in ev_list[0]:
                            eye_event_name = 'FIX'
                            eye_used = ev_list[1]
                            ev_start_sample = float(ev_list[2]) 
                            ev_end_sample = float(ev_list[3]) 
                            dur_in_sec = float(ev_list[4])/sampling_rate
                            x_pos =  float(ev_list[5]) 
                            y_pos =  float(ev_list[6]) 
                            x_pos2 = np.nan
                            y_pos2 = np.nan

                        ## saccades
                        elif 'SAC' in ev_list[0]:
                            eye_event_name = 'SAC'
                            eye_used = ev_list[1]
                            ev_start_sample = float(ev_list[2]) 
                            ev_end_sample = float(ev_list[3]) 
                            dur_in_sec = float(ev_list[4])/sampling_rate
                            x_pos =  float(ev_list[5]) 
                            y_pos =  float(ev_list[6]) 
                            x_pos2 = float(ev_list[7]) 
                            y_pos2 = float(ev_list[8]) 

                        ## concat 
                        eye_events_df = pd.concat((eye_events_df, pd.DataFrame({'trial': [trial_num], 'block_num': [block_num],
                                                            'phase': [phase_num], 'phase_name':[phase_name], 
                                                            'phase_start_sample': [phase_start_sample],
                                                            'eye_event': [eye_event_name], 
                                                            'ev_start_sample': [ev_start_sample], 'ev_end_sample': [ev_end_sample], 
                                                            'dur_in_sec': [dur_in_sec], 'eye': [eye_used],
                                                            'x_pos': [x_pos], 'y_pos': [y_pos], 'x_pos2': [x_pos2], 'y_pos2': [y_pos2]})
                                        ), ignore_index=True)

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

    def __init__(self, behObj):  # initialize child class

        """ Initializes EyeTrack object. 

        Parameters
        ----------
        behObj: object
            task behavior object from  load_beh_data.X 
        """

        # need to initialize parent class (EyeTrack), indicating output infos
        super().__init__(behObj = behObj)

        

class EyeTrackCrowding(EyeTrack):

    def __init__(self, behObj):  # initialize child class

        """ Initializes EyeTrack object. 

        Parameters
        ----------
        behObj: object
            task behavior object from  load_beh_data.X 
        """

        # need to initialize parent class (EyeTrack), indicating output infos
        super().__init__(behObj = behObj)


