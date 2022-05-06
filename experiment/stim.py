import os
import numpy as np
import math
from psychopy import visual, tools

import utils


class Stim(object):

    def __init__(self, session):
        
        """ Initializes a Stim object. 
        Parameters
        ----------
        session : exptools Session object
            A Session object (needed for metadata)
            
        """
        
        # general parameters
        self.session = session


class VsearchStim(Stim):

    def __init__(self, session, grid_pos):

        # need to initialize parent class (Stim)
        super().__init__(session = session)


        self.grid_pos = grid_pos

        # number of elements
        self.nElements = self.grid_pos.shape[0]

        # element positions
        self.element_positions = self.grid_pos

        # element sizes
        self.element_sizes = np.ones((self.nElements)) * self.session.size_pix 

        # elements spatial frequency
        self.element_sfs = np.ones((self.nElements)) * self.session.settings['stimuli']['sf'] # in cycles/gabor width

        # element orientation (half ori1, half ori2)
        self.element_ori = np.ones((self.nElements)) * self.session.settings['stimuli']['ori_deg']

        # element contrasts
        self.element_contrast =  np.ones((self.nElements))

        # element colors 
        self.element_color = np.ones((int(np.round(self.nElements)),3)) * np.array([204, 204, 204])

        # make element array for blue distractors
        self.session.distractors_Bcolor_stim = visual.ElementArrayStim(win = self.session.win, 
                                                                        nElements = self.nElements,
                                                                        units = 'pix', 
                                                                        elementTex = 'sin', 
                                                                        elementMask = 'gauss',
                                                                        sizes = self.element_sizes, 
                                                                        sfs = self.element_sfs, 
                                                                        xys = self.element_positions, 
                                                                        oris = self.element_ori,
                                                                        contrs = self.element_contrast, 
                                                                        colors = self.element_color, 
                                                                        colorSpace = self.session.settings['stimuli']['colorSpace'])
        
        # make element array for pink distractors
        self.session.distractors_Pcolor_stim = visual.ElementArrayStim(win = self.session.win, 
                                                                        nElements = self.nElements,
                                                                        units = 'pix', 
                                                                        elementTex = 'sin', 
                                                                        elementMask = 'gauss',
                                                                        sizes = self.element_sizes, 
                                                                        sfs = self.element_sfs, 
                                                                        xys = self.element_positions, 
                                                                        oris = self.element_ori,
                                                                        contrs = self.element_contrast, 
                                                                        colors = self.element_color, 
                                                                        colorSpace = self.session.settings['stimuli']['colorSpace'])

        # make grating object for target
        self.session.target_stim = visual.ElementArrayStim(win = self.session.win, 
                                                            nElements = self.nElements,
                                                            units = 'pix', 
                                                            elementTex = 'sin', 
                                                            elementMask = 'gauss',
                                                            sizes = self.element_sizes, 
                                                            sfs = self.element_sfs, 
                                                            xys = self.element_positions, 
                                                            oris = self.element_ori,
                                                            contrs = self.element_contrast, 
                                                            colors = self.element_color, 
                                                            colorSpace = self.session.settings['stimuli']['colorSpace'])
        
        
    def draw(self, this_phase, trial_dict):
            
            """ Draw stimuli - pRF bars - for each trial 
            
            Parameters
            ----------
            this_phase: arr
                List/arr of strings with condition names to draw
                
            """


            if this_phase == 'stim':

                ## update blue elements
                self.session.distractors_Bcolor_stim = utils.update_elements(ElementArrayStim = self.session.distractors_Bcolor_stim,
                                                                    elem_positions = trial_dict['distractor_pos'], 
                                                                    grid_pos = self.grid_pos,
                                                                    elem_color = self.session.settings['stimuli']['cond_colors']['blue'],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = trial_dict['distractor_name'],
                                                                    elem_ori = trial_dict['distractor_ori'],
                                                                    key_name = ['bR', 'bL'])

                # update pink elements
                self.session.distractors_Pcolor_stim = utils.update_elements(ElementArrayStim = self.session.distractors_Pcolor_stim,
                                                                    elem_positions = trial_dict['distractor_pos'], 
                                                                    grid_pos = self.grid_pos,
                                                                    elem_color = self.session.settings['stimuli']['cond_colors']['pink'],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = trial_dict['distractor_name'],
                                                                    elem_ori = trial_dict['distractor_ori'],
                                                                    key_name = ['pR', 'pL'])

                
                
                ### NEED TO DEFINE FUNC; DOESNT EXIST YET
                # update target
                self.session.target_stim = utils.update_elements(ElementArrayStim = self.session.target_stim,
                                                                    elem_positions = np.array([list(trial_dict['target_pos'])]), 
                                                                    grid_pos = self.grid_pos,
                                                                    elem_color = trial_dict['target_color'],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = [trial_dict['target_name']],
                                                                    elem_ori = [trial_dict['target_ori']],
                                                                    key_name = [trial_dict['target_name']]) 
                
                
                # actually draw
                self.session.distractors_Bcolor_stim.draw()
                self.session.distractors_Pcolor_stim.draw()
                self.session.target_stim.draw()


