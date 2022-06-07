import os
import os.path as op
import numpy as np

from exptools2.core import Session, PylinkEyetrackerSession

from trial import VsearchTrial, CrowdingTrial #, FlickerTrial
from stim import VsearchStim, CrowdingStim #, FlickerStim

from psychopy import visual, tools
from psychopy.data import QuestHandler, StairHandler

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
        self.gaze_sampleCount = 0
        # radius around fixation (in pix), to check for gaze during iti  
        self.maxDist = 1/utils.dva_per_pix(height_cm = self.settings['monitor_extra']['height'], 
                                                distance_cm = self.settings['monitor']['distance'], 
                                                vert_res_pix = self.screen[-1])

        # target names
        self.target_names = np.array([k for k in self.settings['visual_search']['target_names'].keys()])
        # number of ecc for target
        num_ecc = self.settings['visual_search']['num_ecc']
        # set size to be displayed
        set_size = self.settings['visual_search']['set_size']
        # number of trials per target type
        num_cond_trials = self.settings['visual_search']['num_trl_cond']*len(num_ecc)*len(set_size)

        # make df with trial info, 
        # also including target and distractor positions on screen
        total_trials = 0
        trials_df = pd.DataFrame(columns = ['index', 'block' ,'set_size', 'target_name','target_ecc', 'target_pos', 
                                            'target_color', 'target_ori', 'target_dot', 'target_dot_pos',
                                            'distractor_name', 'distractor_ecc', 'distractor_pos',
                                            'distractor_color', 'distractor_ori', 'distractor_dot', 'distractor_dot_pos'])

        # in each block, target is different
        # but randomize across participants
        self.blk_targets = self.target_names.copy()
        np.random.shuffle(self.blk_targets)

        for blk, name in enumerate(self.blk_targets):

            condition = list(np.tile(name, num_cond_trials))
            condition_ecc = list(np.repeat(num_ecc, num_cond_trials/len(num_ecc)))
            condition_set_size = list(np.tile(set_size, int(num_cond_trials/len(set_size))))

            total_trials += len(condition)

            print('Number of trials for block %i: %i'%(blk,len(condition)))

            # randomize trials
            random_ind = np.arange(len(condition))
            np.random.shuffle(random_ind)
            
            for trial_num, i in enumerate(random_ind):

                # randomly select the position for target
                target_pos_ind = random.choice(np.where(self.grid_ecc == condition_ecc[i])[0])
                
                # and for distractors
                distractor_pos_ind = random.sample([val for val in np.arange(len(self.grid_ecc)) if val != target_pos_ind],
                                                condition_set_size[i]-1)

                # randomly select in which side task dot will be
                # relative to target 
                target_dot = [random.choice(['L','R'])]
                target_dot_pos = self.grid_pos[target_pos_ind].copy()
                
                if target_dot[0] == 'R':
                    target_dot_pos[0] += self.size_pix/2 * self.settings['visual_search']['task_dot_dist']
                else: 
                    target_dot_pos[0] -= self.size_pix/2 * self.settings['visual_search']['task_dot_dist']
                    
                # and to distractors
                distractor_dot_pos = self.grid_pos[distractor_pos_ind].copy()
                distractor_dot = []
                for w in range(condition_set_size[i]-1):
                    
                    distractor_dot.append(random.choice(['L','R']))
                    
                    if distractor_dot[w] == 'R':
                        distractor_dot_pos[w][0] += self.size_pix/2 * self.settings['visual_search']['task_dot_dist']
                    else: 
                        distractor_dot_pos[w][0] -= self.size_pix/2 * self.settings['visual_search']['task_dot_dist']
                
                
                # get distractor names
                dist_name = np.repeat([n for n in self.target_names if n != condition[i]], 
                                    int((condition_set_size[i]-1)/(len(self.target_names)-1)))
                
                
                trials_df = trials_df.append(pd.DataFrame({'index': [trial_num],
                                                        'block': [blk],
                                                        'set_size': [condition_set_size[i]],
                                                        'target_name': [condition[i]],
                                                        'target_ecc': [condition_ecc[i]], 
                                                        'target_pos': [self.grid_pos[target_pos_ind]], 
                                                        'target_color': [self.colors_dict[condition[i]]],
                                                        'target_ori': [self.ori_dict[condition[i]]],
                                                        'target_dot': [target_dot], 
                                                        'target_dot_pos': [target_dot_pos],
                                                        'distractor_name': [dist_name],
                                                        'distractor_ecc': [self.grid_ecc[distractor_pos_ind]],
                                                        'distractor_pos': [self.grid_pos[distractor_pos_ind]],
                                                        'distractor_color': [[self.colors_dict[v] for v in dist_name]], 
                                                        'distractor_ori': [[self.ori_dict[v] for v in dist_name]],
                                                        'distractor_dot': [distractor_dot], 
                                                        'distractor_dot_pos': [distractor_dot_pos]
                                                    }))

        self.total_trials = total_trials
        print('Total number of trials per block: %i'%self.total_trials)

        ## save dataframe with all trial info
        trials_df.to_csv(op.join(self.output_dir, self.output_str+'_trial_info.csv'), index = False)
 
        # max trial time, in seconds
        max_trial_time = self.settings['visual_search']['max_display_time'] + self.settings['visual_search']['max_iti']

        # append all trials
        self.all_trials = []
        for blk in trials_df['block'].unique():

            blk_df = trials_df.loc[trials_df['block']==blk]
    
            for i in np.arange(len(blk_df)):

                # set phase conditions (for logging) and durations
            
                if  i == 0:
                    # insert block phase, to pause trials for a bit
                    # and maybe recalibration of eyetracker
                    phase_cond = tuple(['block_start', 'stim','iti'])
                    phase_dur = tuple([self.settings['visual_search']['max_iti']*100, # make this extremely long 
                                    self.settings['visual_search']['max_display_time'],
                                    self.settings['visual_search']['max_iti']
                                    ])

                else:
                    phase_cond = tuple(['stim','iti'])
                    phase_dur = tuple([self.settings['visual_search']['max_display_time'],
                                    self.settings['visual_search']['max_iti']
                                    ])

                self.all_trials.append(VsearchTrial(session = self ,
                                                    trial_nr = blk_df.iloc[i]['index'],
                                                    phase_durations = phase_dur,
                                                    phase_names = phase_cond,
                                                    trial_dict = blk_df.iloc[i].to_dict(),
                                                    blk_nr = blk
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

        print('Expected number of responses: %d'%(self.total_trials))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses: %d'%self.correct_responses)
        print('Overall accuracy %.2f %%'%(self.correct_responses/self.total_trials*100))

        self.close() # close session

    
class CrowdingSession(ExpSession):
   
    def __init__(self, output_str, output_dir, settings_file, eyetracker_on):  # initialize child class

        """ Initializes CrowdingSession object. 
      
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

        # ecc of target 
        ecc = self.settings['crowding']['ecc']
        self.ecc_pix = ecc/utils.dva_per_pix(height_cm = self.settings['monitor_extra']['height'], 
                                            distance_cm = self.settings['monitor']['distance'], 
                                            vert_res_pix = self.screen[-1])
        # ratio of unflanked trials
        self.unflank_ratio = 1/5

        # number of flankers and position
        self.n_flankers = self.settings['crowding']['num_flankers']

        # distance ratio bounds
        self.distance_ratio_bounds = self.settings['crowding']['staircase']['distance_ratio_bounds']

        # set staircase filename
        self.staircase_file_name = output_str + '_staircase'


    def create_stimuli(self):

        """ Create Stimuli - pRF bars and fixation dot """
        
        #generate PRF stimulus
        self.cwrd_stim = CrowdingStim(session = self)


    def create_trials(self):

        """ Creates trials (before running the session) """

        # some counters for internal bookeeping
        self.total_responses = 0
        self.correct_responses = 0
        self.trial_counter = 0
        self.thisResp = []

        # number of trials per condition
        num_cond_trials = {}
        total_trials = 0
        crowding_type = [] # to store crowding type
        hemifield = [] # to store hemifield
        target_name = [] # target name
                    
        for k in self.settings['crowding']['crwd_type']:
            
            # number of trials for this crowding type
            num_cond_trials[k] = len(self.settings['crowding']['target_names'].keys()) * \
                                self.settings['crowding']['num_trl_cond'] * 2 # num target x min_num trials x 2 hemifields
            # update total number of trials
            total_trials += num_cond_trials[k]

            # set list with crowdign type name, for bookeeping
            crowding_type += list(np.tile(k, num_cond_trials[k]))
            
            # target name
            target_name += list(np.repeat(list(self.settings['crowding']['target_names'].keys()),
                                    self.settings['crowding']['num_trl_cond'] * 2))
            
            # which hemifield we're displaying the stimuli
            hemifield += list(np.tile(['left', 'right'],
                                len(self.settings['crowding']['target_names'].keys()) * \
                                    self.settings['crowding']['num_trl_cond']))
            
        # need to add unflankered trials
        ## need to add same for unflankered trials
        num_cond_trials['unflankered'] = int(total_trials * self.unflank_ratio)
        self.total_trials = total_trials + num_cond_trials['unflankered']

        crowding_type += list(np.tile('unflankered', num_cond_trials['unflankered']))

        target_name += list(np.repeat(list(self.settings['crowding']['target_names'].keys()),
                                    int(num_cond_trials['unflankered']/len(self.settings['crowding']['target_names'].keys()))))

        hemifield += list(np.tile(['left', 'right'],
                                int(num_cond_trials['unflankered']/2)))

        print('Total number of trials: %i'%self.total_trials)

        # now make df with trial info, 
        # also including target and distractor positions on screen
        trials_df = pd.DataFrame(columns = ['index', 'crowding_type', 'hemifield',
                                            'target_name','target_pos', 
                                            'target_color', 'target_ori', 
                                            'distractor_name', 'distractor_pos',
                                            'distractor_color', 'distractor_ori'])

        # randomize trials
        random_ind = np.arange(self.total_trials)
        np.random.shuffle(random_ind)

        for trial_num, i in enumerate(random_ind):

            # stimuli x position
            x_pos_stim = -self.ecc_pix  if hemifield[i] == 'left' else self.ecc_pix

            # set initial distractor position
            dist_pos = utils.get_flanker_pos(num_fl = self.n_flankers, 
                                            offset_ang = self.settings['crowding']['offset_ang'], 
                                            distance_r = self.distance_ratio_bounds[-1], hemi = hemifield[i],
                                            ecc = self.ecc_pix)

            # get distractor (flanker) names
            if crowding_type[i] == 'orientation':

                dist_name = utils.get_flanker_name(target_name[i],
                                                        num_fl = self.n_flankers,
                                                        list_cond = list(self.ori_dict.keys()), 
                                                        same_ori = False, same_color = True)
            elif crowding_type[i] == 'color':

                dist_name = utils.get_flanker_name(target_name[i],
                                                        num_fl = self.n_flankers,
                                                        list_cond = list(self.ori_dict.keys()), 
                                                        same_ori = True, same_color = False)
            elif crowding_type[i] == 'conjunction':

                dist_name = utils.get_flanker_name(target_name[i],
                                                        num_fl = self.n_flankers,
                                                        list_cond = list(self.ori_dict.keys()), 
                                                        same_ori = False, same_color = False)

            else: ## unflankered
                dist_name = list(np.repeat(['None'],self.n_flankers))


            ## append distractor color and orientation
            if crowding_type[i] == 'unflankered':
                dist_col = list(np.repeat(['None'],self.n_flankers))
                dist_ori = list(np.repeat(['None'],self.n_flankers))
            else:
                dist_col = [self.colors_dict[x] for x in dist_name]
                dist_ori = [self.ori_dict[x] for x in dist_name]

            # append trial!
            trials_df = trials_df.append(pd.DataFrame({'index': [trial_num],
                                                    'crowding_type': [crowding_type[i]],
                                                    'hemifield': [hemifield[i]],
                                                    'target_name': [target_name[i]],
                                                    'target_pos': [[x_pos_stim, 0]],
                                                    'target_color': [self.colors_dict[target_name[i]]],
                                                    'target_ori': [self.ori_dict[target_name[i]]],
                                                    'distractor_name':[dist_name],
                                                    'distractor_pos': [dist_pos],
                                                    'distractor_color': [dist_col],
                                                    'distractor_ori': [dist_ori]
                                                }))



        ## save dataframe with all trial info
        trials_df.to_csv(op.join(self.output_dir, self.output_str+'_trial_info.csv'), index = False)

        # trial number to start a new block
        # (this is, introduce a pause and maybe recalibration of eyetracker)
        blk_trials = np.linspace(0, self.total_trials, 
                                self.settings['crowding']['num_blks']+1, dtype=int)[:-1]
        blk_counter = 0

        # append all trials
        self.all_trials = []
        for i in np.arange(self.total_trials):

            # set phase conditions (for logging) and durations

            if blk_trials[blk_counter] == i:
                # insert block phase, to pause trials for a bit
                phase_cond = tuple(['block_start', 'stim', 'response_time','iti'])
                phase_dur = tuple([1000, # make this extremely long
                                self.settings['crowding']['stim_display_time'], 
                                self.settings['crowding']['max_resp_time'], # max time to respond, in seconds
                                self.settings['crowding']['iti']
                                ])

                if blk_counter < self.settings['crowding']['num_blks']-1: 
                    blk_counter += 1

            else:
                phase_cond = tuple(['stim','response_time','iti'])
                phase_dur = tuple([self.settings['crowding']['stim_display_time'],
                                self.settings['crowding']['max_resp_time'],
                                self.settings['crowding']['iti']
                                ])

            self.all_trials.append(CrowdingTrial(session = self ,
                                                trial_nr = trials_df.iloc[i]['index'],
                                                phase_durations = phase_dur,
                                                phase_names = phase_cond,
                                                trial_dict = trials_df.iloc[i].to_dict(),
                                                blk_counter = blk_counter
                                                ))

    
    def create_staircase(self, stair_names = ['orientation', 'color', 'conjunction'],
                            initial_val = .8, minVal = .2, maxVal = .8,
                            pThreshold = 0.83, nUp = 1, nDown = 3, stepSize = 0.05, quest_stair = True):
    
        """ Creates staircases (before running the session) """
        
        self.num_staircases = len(stair_names)

        if isinstance(initial_val, int) or isinstance(initial_val, float):
            self.initial_val = np.tile(initial_val, self.num_staircases)

        elif len(initial_val) < len(self.num_staircases):
            raise ValueError('invalid input, check if initial value of staircases match number of staircases')
        else:
            self.initial_val = initial_val
        
        
        self.staircases = {}
        
        for ind, key in enumerate(stair_names):

            if quest_stair:
                self.staircases[key] = QuestHandler(self.initial_val[ind],
                                                    self.initial_val[ind]*.5,
                                                    pThreshold = pThreshold,
                                                    #nTrials = 20,
                                                    stopInterval = None,
                                                    beta = 3.5,
                                                    delta = 0.05,
                                                    gamma = 0,
                                                    grain = 0.01,
                                                    range = None,
                                                    extraInfo = None,
                                                    minVal = minVal, 
                                                    maxVal = maxVal 
                                                    )
            else: 
                # just make X down - X up staircase
                self.staircases[key] = utils.StaircaseCostum(self.initial_val[ind],
                                                                stepSize = stepSize,
                                                                nUp = nUp, 
                                                                nDown = nDown,
                                                                minVal = minVal, 
                                                                maxVal = maxVal 
                                                                )

    def close_all(self):
        
        """ to guarantee that when closing, everything is saved """

        super(CrowdingSession, self).close()

        for k in self.staircases.keys():
            abs_filename = op.join(self.output_dir, self.staircase_file_name.replace('_staircase', '_staircase_{k}.pickle'.format(k = k)))
            with open(abs_filename, 'wb') as f:
                pickle.dump(self.staircases[k], f)

            #self.staircases[e].saveAsPickle(abs_filename)
            print('Staircase for {name}, has mean {stair_mean}, and standard deviation {stair_std}'.format(name = k, 
                                                                                                        stair_mean = self.staircases[k].mean(), 
                                                                                                        stair_std = self.staircases[k].sd()
                                                                                                        ))
        ## call func to plot staircase outputs
        #print('Accuracy is %.2f %%'%(sum(self.staircases[e].data)/(sum(self.bar_bool)/3)*100))



    def run(self):
        """ Loops over trials and runs them """

        # create trials before running!
        self.create_stimuli()
        self.create_trials()

        # create staircase
        self.create_staircase(stair_names = self.settings['crowding']['crwd_type'],
                            initial_val = self.distance_ratio_bounds[-1], 
                            minVal = self.distance_ratio_bounds[0], 
                            maxVal = self.distance_ratio_bounds[-1],
                            nUp = self.settings['crowding']['staircase']['nUp'], 
                            nDown = self.settings['crowding']['staircase']['nDown'], 
                            stepSize = self.settings['crowding']['staircase']['stepSize'], 
                            quest_stair = self.settings['crowding']['staircase']['quest']) 

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
        this_instruction_string = ('Your task is to indicate\n'
                                'the color and orientation\n'
                                'of the the middle gabor'
                                'by pressing the keys.\n'
                                '\n\n\n'
                                '[Press right index finger\nto continue]\n\n')

        utils.draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'])

        # draw instructions wait a few seconds
        this_instruction_string = ('Sometimes, there will only be one shape presented\n'
                                ' so do not let this confuse you.\n'
                                '\n\n\n'
                                '[Press right index finger\nto continue]\n\n')

        utils.draw_instructions(self.win, this_instruction_string, keys = self.settings['keys']['right_index'])

        # draw instructions wait a few seconds
        this_instruction_string = ('Do NOT look at the shapes!\n'
                                    'Please fixate at the center,\n'
                                    'and do not move your eyes\n'
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

        print('Expected number of responses: %d'%(self.total_trials))
        print('Total subject responses: %d'%self.total_responses)
        print('Correct responses: %d'%self.correct_responses)
        print('Overall accuracy %.2f %%'%(self.correct_responses/self.total_trials*100))

        self.close_all() # close session