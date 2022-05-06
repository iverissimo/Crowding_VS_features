import os
import numpy as np
import yaml, re

from exptools2.core import Trial

from psychopy import event, tools, colors, visual
from psychopy.visual import TextStim

import utils 

import pickle


class VsearchTrial(Trial):

    def __init__(self, session, trial_nr, phase_durations, phase_names, trial_dict, blk_counter = 0,
                 timing = 'seconds', *args, **kwargs):


        """ Initializes a FeatureTrial object. 
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

        """ Draw stimuli - pRF bars - for each trial """
        
        current_time = self.session.clock.getTime() # get time

        if self.phase_names[int(self.phase)] == 'block_start':

            # show instructions
            this_instruction_string = ('BLOCK %i\n\n\n\n\n\n'
                                '[Ready? Press space bar to start]\n\n'%(self.blk_counter+1))

            block_text = visual.TextStim(win = self.session.win, text = this_instruction_string,
                        color = (1, 1, 1), font = 'Helvetica Neue', pos = (0, 0), height = 40,
                        italic = True, alignHoriz = 'center', alignVert = 'center')
    
            # draw text again
            block_text.draw()

        elif self.phase_names[int(self.phase)] == 'stim': 

            # draw target and distractors
            self.session.vs_stim.draw(this_phase = self.phase_names[int(self.phase)],
                                    trial_dict = self.trial_dict)

        ## fixation cross
        self.session.fixation.draw() 

        print(self.phase_names[int(self.phase)])


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

                elif (ev in list(np.ravel(list(self.session.settings['keys']['target_key'].values())))) and \
                    (self.phase_names[int(self.phase)] == 'stim'): # stim presentation

                    event_type = 'response'
                    self.session.total_responses += 1

                    if (ev in self.session.settings['keys']['target_key'][self.trial_dict['target_name']]):
                        self.session.correct_responses += 1
                        print('correct answer')
                    else:
                        print('wrong answer')
                    
                    self.stop_phase()

                else:
                    event_type = 'response'
                    if ev in ['u']: ### TEMPORARY
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

