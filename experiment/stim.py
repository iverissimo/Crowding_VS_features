from multiprocessing.sharedctypes import Value
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

        # task dot size
        self.task_dot_size = np.ones((self.nElements)) * (self.session.size_pix * self.session.settings['visual_search']['task_dot_size'])

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

        # make dot object for task dots
        self.session.dot_stim = visual.ElementArrayStim(win = self.session.win, 
                                                        nElements = self.nElements,
                                                        units = 'pix',
                                                        elementTex = None, 
                                                        elementMask = 'circle',
                                                        sizes = self.task_dot_size,
                                                        xys = self.element_positions)          
        
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

                # update dot positions
                self.session.dot_stim = utils.update_dots(ElementArrayStim = self.session.dot_stim, 
                                                            elem_positions = np.concatenate((trial_dict['target_dot_pos'][np.newaxis,...],
                                                                                            trial_dict['distractor_dot_pos'])), 
                                                            grid_pos = self.grid_pos) 
                
                
                # actually draw
                self.session.distractors_Bcolor_stim.draw()
                self.session.distractors_Pcolor_stim.draw()
                self.session.target_stim.draw()
                self.session.dot_stim.draw()


class CrowdingStim(Stim):

    def __init__(self, session):

        # need to initialize parent class (Stim)
        super().__init__(session = session)
        
        # number of elements
        self.nElements = 1

        # element positions
        self.element_positions = np.array([[self.session.ecc_pix,0]])

        # element sizes
        self.element_sizes = np.ones((self.nElements)) * self.session.size_pix 

        # elements spatial frequency
        self.element_sfs = np.ones((self.nElements)) * self.session.settings['stimuli']['sf'] # in cycles/gabor width

        # element orientation 
        self.element_ori = np.ones((self.nElements)) * self.session.settings['stimuli']['ori_deg']

        # element contrasts
        self.element_contrast =  np.ones((self.nElements))

        # element colors 
        self.element_color = np.ones((int(np.round(self.nElements)),3)) * np.array([204, 204, 204])
        
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
        
        # make grating object for flankers
        ## SHOULD MAKE ONLY 2 OBJECTS -- for now this is quick fix
        self.session.flanker_stim_0 = visual.ElementArrayStim(win = self.session.win, 
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
        
        self.session.flanker_stim_1 = visual.ElementArrayStim(win = self.session.win, 
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

        self.session.flanker_stim_2 = visual.ElementArrayStim(win = self.session.win, 
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
        
        self.session.flanker_stim_3 = visual.ElementArrayStim(win = self.session.win, 
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
        
        
        
    def draw(self, this_phase, trial_dict, spacing_val = 0.8):

        """ Draw stimuli - flankers and target - for each trial 

        Parameters
        ----------
        this_phase: arr
            List/arr of strings with condition names to draw

        """

        if this_phase == 'stim':
            
            # update target
            self.session.target_stim = utils.update_elements(ElementArrayStim = self.session.target_stim,
                                                                elem_positions = trial_dict['target_pos'], 
                                                                elem_color = trial_dict['target_color'],
                                                                elem_sf = self.session.settings['stimuli']['sf'],
                                                                elem_names = [trial_dict['target_name']],
                                                                elem_ori = [trial_dict['target_ori']],
                                                                key_name = [trial_dict['target_name']]) 

            if trial_dict['crowding_type'] != 'unflankered':

                # update y_position of flankers 
                new_positions = utils.get_flanker_pos(num_fl = self.session.n_flankers, 
                                                    offset_ang = self.session.settings['crowding']['offset_ang'], 
                                                    distance_r = spacing_val, 
                                                    hemi = trial_dict['hemifield'],
                                                    ecc = self.session.ecc_pix)

                ## update flankers
                self.session.flanker_stim_0 = utils.update_elements(ElementArrayStim = self.session.flanker_stim_0,
                                                                    elem_positions = new_positions[0], 
                                                                    elem_color = trial_dict['distractor_color'][0],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = [trial_dict['distractor_name'][0]],
                                                                    elem_ori = [trial_dict['distractor_ori'][0]],
                                                                    key_name = [trial_dict['distractor_name'][0]])

                self.session.flanker_stim_1 = utils.update_elements(ElementArrayStim = self.session.flanker_stim_1,
                                                                    elem_positions = new_positions[1],  
                                                                    elem_color = trial_dict['distractor_color'][1],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = [trial_dict['distractor_name'][1]],
                                                                    elem_ori = [trial_dict['distractor_ori'][1]],
                                                                    key_name = [trial_dict['distractor_name'][1]])

                self.session.flanker_stim_2 = utils.update_elements(ElementArrayStim = self.session.flanker_stim_2,
                                                                    elem_positions = new_positions[2],  
                                                                    elem_color = trial_dict['distractor_color'][1],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = [trial_dict['distractor_name'][1]],
                                                                    elem_ori = [trial_dict['distractor_ori'][1]],
                                                                    key_name = [trial_dict['distractor_name'][1]])

                self.session.flanker_stim_3 = utils.update_elements(ElementArrayStim = self.session.flanker_stim_3,
                                                                    elem_positions = new_positions[3],  
                                                                    elem_color = trial_dict['distractor_color'][1],
                                                                    elem_sf = self.session.settings['stimuli']['sf'],
                                                                    elem_names = [trial_dict['distractor_name'][1]],
                                                                    elem_ori = [trial_dict['distractor_ori'][1]],
                                                                    key_name = [trial_dict['distractor_name'][1]])

            # actually draw
            self.session.target_stim.draw()
            
            if trial_dict['crowding_type'] != 'unflankered':
                self.session.flanker_stim_0.draw()
                self.session.flanker_stim_1.draw()
                self.session.flanker_stim_2.draw()
                self.session.flanker_stim_3.draw()