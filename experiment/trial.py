import os
import os.path as op
import numpy as np
import yaml, re

from exptools2.core import Trial

from psychopy import event, tools, colors, visual
from psychopy.visual import TextStim

import utils 

import pickle


class VsearchTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations, phase_names, trial_dict, blk_nr,
                 timing = 'seconds', *args, **kwargs):


        """ Initializes a VsearchTrial object. 
        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
        trial_nr: int
            Trial nr of trial
        timing : str
            The "units" of the phase durations. Default is 'seconds', where we
            assume the phase-durations are in seconds. The other option is
            'frames', where the phase-"duration" refers to the number of frames.
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.session = session

        # phase durations for each condition 
        self.phase_durations = phase_durations
        self.phase_names = phase_names

        # trial dictionary with relevant info
        self.trial_dict = trial_dict
        self.blk_nr = blk_nr

        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)


    def draw(self): 

        """ Draw stimuli - pRF bars - for each trial """
        
        current_time = self.session.clock.getTime() # get time

        if self.phase_names[int(self.phase)] == 'block_start':

            # show instructions
            this_instruction_string = ('BLOCK %i\n\n'
                                        'Search for\n\n\n\n\n\n\n\n'
                                '\n\n[Press space bar to start]\n\n'%((self.blk_nr + 1)))

            block_text = visual.TextStim(win = self.session.win, text = this_instruction_string,
                        color = (1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), height = 40,
                        italic = True, anchorHoriz = 'center', anchorVert = 'center')
    
            # draw text
            block_text.draw()

            img_stim = visual.ImageStim(win = self.session.win, 
                                   image = op.join(os.getcwd(),'instructions_imgs','VS_keys_%s.png'%self.session.blk_targets[self.blk_nr]),
                                   pos = (0, 0))
            img_stim.draw()

        elif self.phase_names[int(self.phase)] == 'stim': 

            # draw target and distractors
            self.session.vs_stim.draw(this_phase = self.phase_names[int(self.phase)],
                                    trial_dict = self.trial_dict)

        else: # ITI period 
            if self.session.eyetracker_on: # if we have eyetracker

                # get current gaze
                curr_gaze = utils.getCurSamp(self.session.tracker, screen = self.session.screen)
                # calculate distance to center of screen
                dist2center = utils.distBetweenPoints(curr_gaze, (0, 0))

                # Check if sample is within boundary
                if dist2center < self.session.maxDist:
                    self.session.gaze_sampleCount += 1

                # If enough samples within boundary
                if self.session.gaze_sampleCount >= 200:
                    print('correctFixation')
                    self.session.gaze_sampleCount = 0
                    self.stop_phase()

        ## fixation cross
        self.session.fixation.draw() 

        #print(self.phase_names[int(self.phase)])


    def get_events(self):
        """ Logs responses/triggers """
        for ev, t in event.getKeys(timeStamped=self.session.clock): # list of of (keyname, time) relative to Clock’s last reset
            if len(ev) > 0:
                if ev in ['q']:
                    print('trial canceled by user')  
                    self.session.close()
                    self.session.quit()

                elif (ev in ['space']) and (self.phase_names[int(self.phase)] == 'block_start'): # new block starts
                    event_type = 'block_start'
                    self.stop_phase()

                else:
                    event_type = 'response'

                    if (ev in np.concatenate((self.session.settings['keys']['left_index'], self.session.settings['keys']['right_index']))) and \
                        (self.phase_names[int(self.phase)] == 'stim'): # stim presentation

                        self.session.total_responses += 1

                        if (ev in self.session.settings['keys']['left_index']) and \
                            (self.trial_dict['target_dot'][0] == 'L'):

                            self.session.correct_responses += 1
                            print('correct answer')

                        elif (ev in self.session.settings['keys']['right_index']) and \
                            (self.trial_dict['target_dot'][0] == 'R'):

                            self.session.correct_responses += 1
                            print('correct answer')

                        else:
                            print('wrong answer')
                        
                        self.stop_phase()
                    

                # log everything into session data frame
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.ID
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev                

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val


class CrowdingTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations, phase_names, trial_dict, blk_counter = 0,
                 timing = 'seconds', *args, **kwargs):


        """ Initializes a CrowdingTrial object. 
        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
        trial_nr: int
            Trial nr of trial
        timing : str
            The "units" of the phase durations. Default is 'seconds', where we
            assume the phase-durations are in seconds. The other option is
            'frames', where the phase-"duration" refers to the number of frames.
        """
        
        self.ID = trial_nr # trial identifier, not sure if needed
        self.session = session

        # phase durations for each condition 
        self.phase_durations = phase_durations
        self.phase_names = phase_names

        # trial dictionary with relevant info
        self.trial_dict = trial_dict
        self.blk_counter = blk_counter

        super().__init__(session, trial_nr, phase_durations, phase_names, verbose=False, *args, **kwargs)

    
    def draw(self): 

        """ Draw stimuli - target and flankers - for each trial """

        current_time = self.session.clock.getTime() # get time

        if self.phase_names[int(self.phase)] == 'block_start':

            # show instructions
            if self.blk_counter == 0:
                this_instruction_string = ('BLOCK %i\n\n\n\n\n\n'
                                '\n\n\n'%(self.blk_counter + 1))
            else:
                this_instruction_string = ('BLOCK %i\n\n\n\n\n\n'
                                    '[Press space bar to start]\n\n'%(self.blk_counter + 1))

            block_text = visual.TextStim(win = self.session.win, text = this_instruction_string,
                        color = (1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), height = 40,
                        italic = True, anchorHoriz = 'center', anchorVert = 'center')

            # draw text again
            block_text.draw()

        elif self.phase_names[int(self.phase)] == 'stim':

            # define spacing ratio given staircase
            if self.trial_dict['crowding_type'] == 'unflankered':
                spacing_val = 0
            else:
                if self.session.settings['crowding']['staircase']['quest']:
                    spacing_val = np.clip(self.session.staircases[self.trial_dict['crowding_type']].quantile(), 
                                    self.session.distance_ratio_bounds[0], 
                                    self.session.distance_ratio_bounds[1])
                else:
                    spacing_val = np.clip(self.session.staircases[self.trial_dict['crowding_type']]._nextIntensity,
                                    self.session.distance_ratio_bounds[0], 
                                    self.session.distance_ratio_bounds[1]) 

            # draw target and distractors
            self.session.cwrd_stim.draw(this_phase = self.phase_names[int(self.phase)],
                                        trial_dict = self.trial_dict,
                                        spacing_val = spacing_val)

        elif self.phase_names[int(self.phase)] == 'iti': # iti
            
            if self.session.trial_counter <= self.ID: # if no response was given before

                user_response = 0
                if self.trial_dict['crowding_type'] != 'unflankered':
                    # update staircase
                    self.session.staircases[self.trial_dict['crowding_type']].addResponse(user_response)

                self.session.trial_counter += 1 # update trial counter 

        ## fixation cross
        self.session.fixation.draw() 

        #print(self.phase_names[int(self.phase)])


    def get_events(self):
        """ Logs responses/triggers """
        for ev, t in event.getKeys(timeStamped=self.session.clock): # list of of (keyname, time) relative to Clock’s last reset
            if len(ev) > 0:
                if ev in ['q']:
                    print('trial canceled by user')  
                    self.session.close_all()
                    self.session.quit()

                elif (ev in ['space']) and (self.phase_names[int(self.phase)] == 'block_start'): # new block starts
                    event_type = 'block_start'
                    self.stop_phase()

                else:
                    event_type = 'response'
                    self.session.total_responses += 1

                    if self.phase_names[int(self.phase)] == 'response_time':
                    
                        if (ev in list(np.ravel(list(self.session.settings['keys']['target_key'].values())))): 

                            if self.session.trial_counter <= self.ID:

                                ## get user response!
                                user_response = utils.get_response4staircase(event_key = ev, 
                                                                        target_key = self.session.settings['keys']['target_key'][self.trial_dict['target_name']])

                                self.session.thisResp.append(user_response)
                                self.session.correct_responses += user_response

                                # update color with answer
                                if len(self.session.thisResp) > 0: # update with answer
                                    if self.trial_dict['crowding_type'] != 'unflankered':
                                        # update staircase
                                        self.session.staircases[self.trial_dict['crowding_type']].addResponse(self.session.thisResp[-1])
                                    # reset response again
                                    self.session.thisResp = []

                                self.session.trial_counter += 1 # update trial counter   

                            self.stop_phase()

                # log everything into session data frame
                idx = self.session.global_log.shape[0]
                self.session.global_log.loc[idx, 'trial_nr'] = self.ID
                self.session.global_log.loc[idx, 'onset'] = t
                self.session.global_log.loc[idx, 'event_type'] = event_type
                self.session.global_log.loc[idx, 'phase'] = self.phase
                self.session.global_log.loc[idx, 'response'] = ev                

                for param, val in self.parameters.items():
                    self.session.global_log.loc[idx, param] = val

