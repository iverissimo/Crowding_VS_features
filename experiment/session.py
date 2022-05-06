import os
import os.path as op
import numpy as np

from exptools2.core import Session, PylinkEyetrackerSession

from trial import VsearchTrial #, CrowdingTrial, FlickerTrial
from stim import VsearchStim #, CrowdingStim, FlickerStim

from psychopy import visual, tools

import itertools
import pickle

import utils

import random
import pandas as pd


class ExpSession(PylinkEyetrackerSession):

    def __init__(self, output_str, output_dir, settings_file, eyetracker_on = True):  # initialize child class

            """ Initializes ExpSession object. 
          
            Parameters
            ----------
            output_str : str
                Basename for all output-files (like logs), e.g., "sub-01_task-PRFstandard_run-1"
            output_dir : str
                Path to desired output-directory (default: None, which results in $pwd/logs)
            settings_file : str
                Path to yaml-file with settings (default: None, which results in the package's
                default settings file (in data/default_settings.yml)
            """

            # need to initialize parent class (Session), indicating output infos
            super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, eyetracker_on = eyetracker_on)

            # set size of display
            self.screen = np.array([self.win.size[0], self.win.size[1]])

            if self.settings['window_extra']['mac_bool']: # to compensate for macbook retina display
                self.screen = self.screen/2
                print('Running experiment on macbook, defining display accordingly')

            print('Screen res is %s'%str(self.screen ))

            ## some general parameters

            # get element size
            size_deg = self.settings['stimuli']['size_deg']
            self.size_pix = size_deg/utils.dva_per_pix(height_cm = self.settings['monitor_extra']['height'], 
                                                distance_cm = self.settings['monitor']['distance'], 
                                                vert_res_pix = self.screen[-1])
            print('gabor diameter in pix %s'%str(self.size_pix))

            # set condition colors and orientation values
            self.colors_dict = {}
            self.ori_dict = {}

            for k in self.settings['visual_search']['target_names'].keys():
                
                self.colors_dict[k] = self.settings['stimuli']['cond_colors']['blue'] if 'b' in k else self.settings['stimuli']['cond_colors']['pink']
                self.ori_dict[k] = self.settings['stimuli']['ori_deg'] if 'R' in k else 360-self.settings['stimuli']['ori_deg']


            ## create some elements that will be common to both tasks ##
            linesize_pix = self.settings['stimuli']['fix_lineSize_deg']/utils.dva_per_pix(height_cm = self.settings['monitor_extra']['height'], 
                                                                                        distance_cm = self.settings['monitor']['distance'], 
                                                                                        vert_res_pix = self.screen[-1])
            linewidth_pix = self.settings['stimuli']['fix_linewidth_deg']/utils.dva_per_pix(height_cm = self.settings['monitor_extra']['height'], 
                                                                                        distance_cm = self.settings['monitor']['distance'], 
                                                                                        vert_res_pix = self.screen[-1])            
            self.fixation = visual.ShapeStim(self.win, 
                                            vertices = ((0, -linesize_pix/2), (0, linesize_pix/2), 
                                                        (0,0), 
                                                        (-linesize_pix/2, 0), (linesize_pix/2, 0)),
                                            lineWidth = linewidth_pix,
                                            closeShape = False,
                                            lineColor = self.settings['stimuli']['fix_color'])
                        

            
class VsearchSession(ExpSession):
   
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on):  # initialize child class

        """ Initializes VsearchSession object. 
      
        Parameters
        ----------
        output_str : str
            Basename for all output-files (like logs), e.g., "sub-01_task-PRFstandard_run-1"
        output_dir : str
            Path to desired output-directory (default: None, which results in $pwd/logs)
        settings_file : str
            Path to yaml-file with settings (default: None, which results in the package's
            default settings file (in data/default_settings.yml)
        """

        # need to initialize parent class (ExpSession), indicating output infos
        super().__init__(output_str = output_str, output_dir = output_dir, settings_file = settings_file, 
                        eyetracker_on = eyetracker_on)

        ## get grid positions where stimuli can be 

        # range of possible ecc
        ecc_range = np.arange(self.settings['visual_search']['min_ecc'], 
                            self.settings['visual_search']['max_ecc'] + self.settings['visual_search']['ecc_grid_step'], 
                            self.settings['visual_search']['ecc_grid_step'])
        # number of points per ecc
        n_points = [(i+1)*4 for i, _ in enumerate(ecc_range)] # increasing 4 per circle, maybe choose density differently?

        # define initial circle grid for positions
        circles = utils.circle_points(ecc_range, n_points)

        # constrain them within ellipse - necessary? 
        # and get actual position and ecc list
        self.grid_pos, self.grid_ecc = utils.get_grid_array(circles, ecc_range, 
                                                    convert2pix = True, screen = self.screen, 
                                                    height_cm = self.settings['monitor_extra']['height'], 
                                                    distance_cm = self.settings['monitor']['distance'], 
                                                    constraint_type = 'ellipse', 
                                                    constraint_bounds_pix = self.screen/2 - self.size_pix/2)

        print('Possible grid positions %i'%self.grid_pos.shape[0])

    
    def create_stimuli(self):

        """ Create Stimuli - pRF bars and fixation dot """
        
        #generate PRF stimulus
        self.vs_stim = VsearchStim(session = self,  
                                    grid_pos = self.grid_pos
                                    )

    def create_trials(self):

        """ Creates trials (before running the session) """

        # some counters for internal bookeeping
        self.total_responses = 0
        self.correct_responses = 0

        # target names
        self.target_names = np.array([k for k in self.settings['visual_search']['target_names'].keys()])
        # number of ecc for target
        num_ecc = self.settings['visual_search']['num_ecc']
        # set size to be displayed
        set_size = self.settings['visual_search']['set_size']
        # number of trials per target type
        num_cond_trials = self.settings['visual_search']['num_trl_cond']*len(num_ecc)*len(set_size)

        # make arrays with relevant info for total amount of trials
        condition = []
        condition_ecc = []
        condition_set_size = []
        #condition_pos = []

        for name in self.target_names:
            
            condition += list(np.tile(name, num_cond_trials))
            condition_ecc += list(np.repeat(num_ecc, num_cond_trials/len(num_ecc)))
            condition_set_size += list(np.repeat(set_size, num_cond_trials/len(set_size)))

        self.total_trials = len(condition)
        print('Total number of trials: %i'%self.total_trials)


        # now make df with trial info, 
        # also including target and distractor positions on screen
        trials_df = pd.DataFrame(columns = ['index', 'set_size', 'target_name','target_ecc', 'target_pos', 
                                            'target_color', 'target_ori', 
                                            'distractor_name', 'distractor_ecc', 'distractor_pos',
                                            'distractor_color', 'distractor_ori'])

        # randomize trials
        random_ind = np.arange(self.total_trials)
        np.random.shuffle(random_ind)

        for trial_num, i in enumerate(random_ind):
            
            # randomly select the position for target
            target_pos_ind = random.choice(np.where(self.grid_ecc == condition_ecc[i])[0])
            
            # and for distractors
            distractor_pos_ind = random.sample([val for val in np.arange(len(self.grid_ecc)) if val != target_pos_ind],
                                            condition_set_size[i]-1)
            
            # get distractor names
            dist_name = np.repeat([n for n in self.target_names if n != condition[i]], 
                                int((condition_set_size[i]-1)/(len(self.target_names)-1)))
            
            
            trials_df = trials_df.append(pd.DataFrame({'index': [trial_num],
                                                    'set_size': [condition_set_size[i]],
                                                    'target_name': [condition[i]],
                                                    'target_ecc': [condition_ecc[i]], 
                                                    'target_pos': [self.grid_pos[target_pos_ind]], 
                                                    'target_color': [self.colors_dict[condition[i]]],
                                                    'target_ori': [self.ori_dict[condition[i]]],
                                                    'distractor_name': [dist_name],
                                                    'distractor_ecc': [self.grid_ecc[distractor_pos_ind]],
                                                    'distractor_pos': [self.grid_pos[distractor_pos_ind]],
                                                    'distractor_color': [[self.colors_dict[v] for v in dist_name]], 
                                                    'distractor_ori': [[self.ori_dict[v] for v in dist_name]]
                                                }))

        ## save dataframe with all trial info
        trials_df.to_csv(op.join(self.output_dir, self.output_str+'_trial_info.csv'), index = False)

        # trial number to start a new block
        # (this is, introduce a pause and maybe recalibration of eyetracker)
        blk_trials = np.linspace(0, self.total_trials, 
                                self.settings['visual_search']['num_blks']+1, dtype=int)[:-1]
        blk_counter = 0
 
        # max trial time, in seconds
        max_trial_time = self.settings['visual_search']['max_display_time'] + self.settings['visual_search']['max_iti']

        # append all trials
        self.all_trials = []
        for i in np.arange(self.total_trials):

            # set phase conditions (for logging) and durations
            
            if blk_trials[blk_counter] == i:
                # insert block phase, to pause trials for a bit
                phase_cond = tuple(['block_start', 'stim','iti'])
                phase_dur = tuple([self.settings['visual_search']['max_iti']*100, # make this extremely long 
                                self.settings['visual_search']['max_display_time'],
                                self.settings['visual_search']['max_iti']
                                ])
                
                if blk_counter < self.settings['visual_search']['num_blks']-1: 
                    blk_counter += 1

            else:
                phase_cond = tuple(['stim','iti'])
                phase_dur = tuple([self.settings['visual_search']['max_display_time'],
                                self.settings['visual_search']['max_iti']
                                ])

            self.all_trials.append(VsearchTrial(session = self ,
                                                trial_nr = trials_df.iloc[i]['index'],
                                                phase_durations = phase_dur,
                                                phase_names = phase_cond,
                                                trial_dict = trials_df.iloc[i].to_dict(),
                                                blk_counter = blk_counter-1
                                                ))

    def run(self):
        """ Loops over trials and runs them """

        # create trials before running!
        self.create_stimuli()
        self.create_trials()

        # if eyetracking then calibrate
        if self.eyetracker_on:
            self.calibrate_eyetracker()

        # draw instructions wait a few seconds
        this_instruction_string = ('During the experiment\nyou will see several gabors.\n\n'
                                'They can be pink or blue,\n'
                                'and be tilted to the right or left\n'
                                '\n\n\n'
                                '[Press right index finger\nto continue]\n\n')

        utils.draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'])

        # draw instructions wait a few seconds
        this_instruction_string = ('Your task is to find\nthe UNIQUE gabor\n'
                                'and indicate its color and orientation\n'
                                'by pressing the keys.\n'
                                '\n\n\n'
                                '[Press right index finger\nto continue]\n\n')

        utils.draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'])

        # draw instructions wait a few seconds
        this_instruction_string = ('You can move your eyes around\n'
                                'to search for your target.\n\n'
                                'Please return to the fixation cross\nat the end of each trial.\n'
                                '\n\n\n'
                                '[Ready? Press space bar to start]\n\n')

        utils.draw_instructions(self.win, this_instruction_string, keys = ['space'])

        # start recording gaze
        if self.eyetracker_on:
            self.start_recording_eyetracker()

        self.start_experiment()
        
        # cycle through trials
        for trl in self.all_trials: 
            trl.run() # run forrest run

        self.close() # close session

    
    
